"""
Supervisor for multi-agent orchestration.
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.agents.critic_agent import CriticAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.utils.logger import get_logger

logger = get_logger()


class SupervisorState(TypedDict):
    """State for multi-agent supervisor workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    next_agent: Literal["retriever", "summarizer", "generator", "critic", "END"] | None
    documents: list[Document]
    summary: str
    answer: str
    critique: dict
    iteration: int
    max_iterations: int


class MultiAgentSupervisor:
    """
    Orchestrates specialized agents for RAG.

    Workflow:
    1. Retriever fetches documents
    2. Summarizer condenses relevant content
    3. Generator creates answer
    4. Critic validates quality
    5. Iterate if revision needed
    """

    def __init__(self, vectorstore, llm, checkpointer=None, max_iterations: int = 3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.checkpointer = checkpointer
        self.max_iterations = max_iterations

        self.summarizer = SummarizerAgent(llm)
        self.critic = CriticAgent(llm)

        self.routing_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a workflow coordinator. Based on the current state, decide the next step:

- retriever: Need to fetch documents (no documents yet)
- summarizer: Have documents, need to condense them (no summary yet)
- generator: Have summary/documents, need to generate answer (no answer yet)
- critic: Have answer, need to validate quality
- END: Task is complete (have validated answer or max iterations reached)

Respond with ONLY one word: retriever, summarizer, generator, critic, or END""",
                ),
                (
                    "human",
                    """Current State:
- Question: {question}
- Has Documents: {has_docs}
- Has Summary: {has_summary}
- Has Answer: {has_answer}
- Iteration: {iteration}/{max_iter}
- Last Critique: {critique}

What's the next step?""",
                ),
            ]
        )

        self.graph = self._build_graph()
        logger.info("MultiAgentSupervisor initialized")

    async def supervisor_routing(self, state: SupervisorState) -> SupervisorState:
        """Decide which agent to call next."""
        # Simple rule-based routing for reliability
        if len(state["documents"]) == 0:
            state["next_agent"] = "retriever"
        elif not state["summary"] and len(state["documents"]) > 0:
            state["next_agent"] = "summarizer"
        elif not state["answer"]:
            state["next_agent"] = "generator"
        elif state["iteration"] < state["max_iterations"] and not state.get("critique"):
            state["next_agent"] = "critic"
        elif state["critique"].get("needs_revision") and state["iteration"] < state["max_iterations"]:
            state["next_agent"] = "generator"
        else:
            state["next_agent"] = "END"

        logger.info(f"Supervisor routing to: {state['next_agent']}")
        return state

    def route_to_agent(self, state: SupervisorState) -> Literal["retriever", "summarizer", "generator", "critic", "END"]:
        """Return next agent based on state."""
        next_agent = state.get("next_agent", "END")
        return next_agent if next_agent in ["retriever", "summarizer", "generator", "critic", "END"] else "END"  # type: ignore

    async def call_retriever(self, state: SupervisorState) -> SupervisorState:
        """Retrieve documents from vectorstore."""
        logger.info(f"Retriever: fetching documents for '{state['question']}'")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = await retriever.ainvoke(state["question"])

        state["documents"] = docs
        state["iteration"] += 1

        logger.info(f"Retriever: found {len(docs)} documents")
        return state

    async def call_summarizer(self, state: SupervisorState) -> SupervisorState:
        """Summarize retrieved documents."""
        logger.info(f"Summarizer: condensing {len(state['documents'])} documents")

        summary = await self.summarizer.summarize(state["documents"], state["question"])
        state["summary"] = summary

        logger.info(f"Summarizer: generated summary of {len(summary)} chars")
        return state

    async def generate_answer(self, state: SupervisorState) -> SupervisorState:
        """Generate answer from documents/summary."""
        logger.info("Generator: creating answer")

        # Use summary if available, otherwise use raw documents
        context = state["summary"] or "\n\n".join([d.page_content for d in state["documents"]])

        # Include critique feedback if revising
        revision_context = ""
        if state["critique"].get("needs_revision"):
            issues = state["critique"].get("issues", [])
            suggestions = state["critique"].get("suggestions", [])
            if issues or suggestions:
                revision_context = f"""
Previous answer had issues. Please address:
Issues: {', '.join(issues) if issues else 'None'}
Suggestions: {', '.join(suggestions) if suggestions else 'None'}

"""

        prompt = f"""{revision_context}Based on the following context, answer the question.

Context:
{context}

Question: {state['question']}

Provide a comprehensive answer:"""

        response = await self.llm.ainvoke(prompt)
        state["answer"] = response.content if hasattr(response, "content") else str(response)

        logger.info(f"Generator: created answer of {len(state['answer'])} chars")
        return state

    async def call_critic(self, state: SupervisorState) -> SupervisorState:
        """Validate answer quality."""
        logger.info("Critic: evaluating answer")

        docs_text = "\n\n".join([d.page_content for d in state["documents"]])
        critique = await self.critic.critique(state["question"], state["answer"], docs_text)

        state["critique"] = {
            "accuracy": critique.accuracy_score,
            "completeness": critique.completeness_score,
            "clarity": critique.clarity_score,
            "needs_revision": critique.needs_revision,
            "issues": critique.issues,
            "suggestions": critique.suggestions,
            "overall_score": self.critic.get_overall_score(critique),
        }

        logger.info(f"Critic: overall score {state['critique']['overall_score']}, " f"needs_revision={critique.needs_revision}")

        # If revision needed and we have iterations left, clear the answer to regenerate
        if critique.needs_revision and state["iteration"] < state["max_iterations"]:
            state["iteration"] += 1
            logger.info(f"Critic: requesting revision (iteration {state['iteration']})")

        return state

    async def finalize(self, state: SupervisorState) -> SupervisorState:
        """Finalize the workflow and add messages."""
        logger.info("Finalizing multi-agent workflow")

        state["messages"].append(HumanMessage(content=state["question"]))
        state["messages"].append(AIMessage(content=state["answer"]))

        return state

    def _build_graph(self):
        """Build the multi-agent workflow graph."""
        workflow = StateGraph(SupervisorState)

        # Add nodes
        workflow.add_node("supervisor", self.supervisor_routing)
        workflow.add_node("retriever", self.call_retriever)
        workflow.add_node("summarizer", self.call_summarizer)
        workflow.add_node("generator", self.generate_answer)
        workflow.add_node("critic", self.call_critic)
        workflow.add_node("finalize", self.finalize)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self.route_to_agent,
            {"retriever": "retriever", "summarizer": "summarizer", "generator": "generator", "critic": "critic", "END": "finalize"},
        )

        # All agents return to supervisor for next decision
        workflow.add_edge("retriever", "supervisor")
        workflow.add_edge("summarizer", "supervisor")
        workflow.add_edge("generator", "supervisor")
        workflow.add_edge("critic", "supervisor")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def invoke(self, question: str, config: dict[Any, Any] | None = None) -> str:
        """Run the multi-agent workflow."""
        initial_state: SupervisorState = {
            "messages": [],
            "question": question,
            "next_agent": None,
            "documents": [],
            "summary": "",
            "answer": "",
            "critique": {},
            "iteration": 0,
            "max_iterations": self.max_iterations,
        }

        result = await self.graph.ainvoke(initial_state, config=config)
        return result["answer"]

    def get_workflow_summary(self, state: SupervisorState) -> dict:
        """Get summary of workflow execution."""
        return {
            "question": state["question"],
            "documents_retrieved": len(state["documents"]),
            "summary_length": len(state["summary"]) if state["summary"] else 0,
            "answer_length": len(state["answer"]) if state["answer"] else 0,
            "iterations": state["iteration"],
            "final_score": state["critique"].get("overall_score", None),
            "critique": state["critique"],
        }
