"""
Corrective RAG (CRAG) implementation using LangGraph.

Workflow:
1. Retrieve documents from vectorstore
2. Grade relevance of each document
3. If sufficient relevant docs -> Generate answer
4. If insufficient -> Web search -> Generate answer
5. Check for hallucinations
6. Return validated answer
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.evaluation.rag_evaluator import RAGEvaluator
from src.tools.web_search import web_search_to_documents
from src.utils.logger import get_logger

logger = get_logger()


class CRAGState(TypedDict):
    """State for Corrective RAG workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    relevant_docs: list[Document]
    web_search_needed: bool
    web_results: list[Document]
    generation: str
    is_grounded: bool


class CorrectiveRAGGraph:
    """
    Self-corrective RAG with relevance grading and web search fallback.
    """

    def __init__(self, vectorstore, llm, checkpointer=None, relevance_threshold: float = 0.5, min_relevant_docs: int = 1):
        self.vectorstore = vectorstore
        self.llm = llm
        self.checkpointer = checkpointer
        self.evaluator = RAGEvaluator(llm)
        self.relevance_threshold = relevance_threshold
        self.min_relevant_docs = min_relevant_docs

        self.graph = self._build_graph()
        logger.info("CorrectiveRAGGraph initialized")

    async def retrieve(self, state: CRAGState) -> CRAGState:
        """Retrieve documents from vectorstore."""
        logger.info(f"Retrieving documents for: {state['question']}")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = await retriever.ainvoke(state["question"])

        state["documents"] = docs
        logger.info(f"Retrieved {len(docs)} documents")
        return state

    async def grade_documents(self, state: CRAGState) -> CRAGState:
        """Grade relevance of retrieved documents."""
        logger.info("Grading document relevance")

        relevant_docs = []
        for doc in state["documents"]:
            score = await self.evaluator.grade_relevance(state["question"], doc.page_content)
            if score.score >= self.relevance_threshold:
                relevant_docs.append(doc)

        state["relevant_docs"] = relevant_docs
        state["web_search_needed"] = len(relevant_docs) < self.min_relevant_docs

        logger.info(f"Relevant: {len(relevant_docs)}, Web search: {state['web_search_needed']}")
        return state

    def decide_retrieval_quality(self, state: CRAGState) -> Literal["generate", "web_search"]:
        """Route based on document relevance."""
        if state["web_search_needed"]:
            return "web_search"
        return "generate"

    async def web_search(self, state: CRAGState) -> CRAGState:
        """Fallback to web search for additional context."""
        logger.info(f"Web search for: {state['question']}")

        web_docs = web_search_to_documents(state["question"], max_results=3)
        state["web_results"] = web_docs
        state["relevant_docs"].extend(web_docs)

        logger.info(f"Added {len(web_docs)} web results")
        return state

    async def generate(self, state: CRAGState) -> CRAGState:
        """Generate answer from relevant documents."""
        logger.info("Generating answer")

        context = "\n\n".join([doc.page_content for doc in state["relevant_docs"]])

        prompt = f"""Answer the question based on the following context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {state['question']}

Answer:"""

        response = await self.llm.ainvoke(prompt)
        state["generation"] = response.content if hasattr(response, "content") else str(response)

        state["messages"].append(HumanMessage(content=state["question"]))
        state["messages"].append(AIMessage(content=state["generation"]))

        return state

    async def check_hallucination(self, state: CRAGState) -> CRAGState:
        """Check if answer is grounded in documents."""
        logger.info("Checking for hallucinations")

        context = "\n\n".join([doc.page_content for doc in state["relevant_docs"]])
        check = await self.evaluator.check_hallucination(context, state["generation"])

        state["is_grounded"] = check.is_grounded

        if not check.is_grounded:
            logger.warning(f"Hallucination detected: {check.unsupported_claims}")

        return state

    def _build_graph(self):
        """Build the CRAG workflow graph."""
        workflow = StateGraph(CRAGState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate", self.generate)
        workflow.add_node("check_hallucination", self.check_hallucination)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges("grade_documents", self.decide_retrieval_quality, {"generate": "generate", "web_search": "web_search"})

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", "check_hallucination")
        workflow.add_edge("check_hallucination", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def invoke(self, question: str, config: dict[Any, Any] | None = None) -> str:
        """Run the CRAG workflow."""
        initial_state = {
            "messages": [],
            "question": question,
            "documents": [],
            "relevant_docs": [],
            "web_search_needed": False,
            "web_results": [],
            "generation": "",
            "is_grounded": True,
        }

        result = await self.graph.ainvoke(initial_state, config=config)
        return result["generation"]
