"""
Adaptive RAG with query routing and strategy selection.

Routes queries to optimal retrieval strategy:
- Simple: Direct vectorstore retrieval (fast)
- Complex: Multi-step retrieval with reranking
- Web: External web search for out-of-domain queries
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.tools.web_search import web_search_to_documents
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger()


class AdaptiveRAGState(TypedDict):
    """State for Adaptive RAG workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    query_type: Literal["simple", "complex", "web"] | None
    documents: list[Document]
    generation: str


class AdaptiveRAGGraph:
    """
    Adaptive RAG with intelligent query routing.

    Supports hybrid retrieval when a HybridRetriever is provided.
    """

    def __init__(
        self,
        vectorstore,
        llm,
        retriever: HybridRetriever | None = None,
        checkpointer=None,
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = retriever
        self.checkpointer = checkpointer

        self.classification_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Classify the query type:

- simple: Direct factual question, single document lookup
- complex: Requires reasoning or multiple documents
- web: Current events or external information

Respond with ONLY: simple, complex, or web""",
                ),
                ("human", "{question}"),
            ]
        )

        self.graph = self._build_graph()
        logger.info(f"AdaptiveRAGGraph initialized (hybrid_retriever={retriever is not None})")

    async def classify_query(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Classify query complexity/type."""
        logger.info(f"Classifying: {state['question']}")

        chain = self.classification_prompt | self.llm
        response = await chain.ainvoke({"question": state["question"]})

        classification = response.content.strip().lower() if hasattr(response, "content") else "simple"

        if classification not in ["simple", "complex", "web"]:
            classification = "simple"

        state["query_type"] = classification  # type: ignore
        logger.info(f"Classified as: {classification}")
        return state

    def route_query(self, state: AdaptiveRAGState) -> Literal["simple", "complex", "web"]:
        """Route to appropriate retrieval strategy."""
        query_type = state.get("query_type", "simple")
        return query_type if query_type in ["simple", "complex", "web"] else "simple"  # type: ignore

    async def simple_retrieval(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Fast direct retrieval for simple queries."""
        logger.info("Simple retrieval")

        if self.retriever:
            docs = await self.retriever.ainvoke(state["question"], k=3)
            logger.debug("Using HybridRetriever for simple retrieval")
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = await retriever.ainvoke(state["question"])

        state["documents"] = docs
        return state

    async def complex_retrieval(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Multi-step retrieval for complex queries."""
        logger.info("Complex retrieval")

        if self.retriever:
            initial_docs = await self.retriever.ainvoke(state["question"], k=6)
            logger.debug("Using HybridRetriever for complex retrieval")
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
            initial_docs = await retriever.ainvoke(state["question"])

        # Generate sub-queries
        subquery_prompt = ChatPromptTemplate.from_messages([("system", "Generate 2 related queries. One per line."), ("human", "{question}")])

        chain = subquery_prompt | self.llm
        response = await chain.ainvoke({"question": state["question"]})

        subqueries = response.content.strip().split("\n") if hasattr(response, "content") else []

        all_docs = list(initial_docs)
        for sq in subqueries[:2]:
            if sq.strip():
                if self.retriever:
                    sub_docs = await self.retriever.ainvoke(sq.strip(), k=3)
                else:
                    retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                    sub_docs = await retriever.ainvoke(sq.strip())
                all_docs.extend(sub_docs)

        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        state["documents"] = unique_docs[:8]
        return state

    async def web_retrieval(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Web search for external information."""
        logger.info("Web retrieval")

        if self.retriever:
            local_docs = await self.retriever.ainvoke(state["question"], k=2)
            logger.debug("Using HybridRetriever for web retrieval")
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
            local_docs = await retriever.ainvoke(state["question"])

        web_docs = web_search_to_documents(state["question"], max_results=3)

        state["documents"] = local_docs + web_docs
        return state

    async def generate(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Generate answer."""
        logger.info("Generating")

        context = "\n\n".join([doc.page_content for doc in state["documents"]])

        prompt = f"""Context:
{context}

Question: {state['question']}

Answer:"""

        response = await self.llm.ainvoke(prompt)
        state["generation"] = response.content if hasattr(response, "content") else str(response)

        state["messages"].append(HumanMessage(content=state["question"]))
        state["messages"].append(AIMessage(content=state["generation"]))

        return state

    def _build_graph(self):
        """Build workflow graph."""
        workflow = StateGraph(AdaptiveRAGState)

        workflow.add_node("classify", self.classify_query)
        workflow.add_node("simple_retrieval", self.simple_retrieval)
        workflow.add_node("complex_retrieval", self.complex_retrieval)
        workflow.add_node("web_retrieval", self.web_retrieval)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("classify")

        workflow.add_conditional_edges(
            "classify", self.route_query, {"simple": "simple_retrieval", "complex": "complex_retrieval", "web": "web_retrieval"}
        )

        workflow.add_edge("simple_retrieval", "generate")
        workflow.add_edge("complex_retrieval", "generate")
        workflow.add_edge("web_retrieval", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def invoke(self, question: str, config: dict[Any, Any] | None = None) -> str:
        """Run workflow."""
        initial_state: AdaptiveRAGState = {"messages": [], "question": question, "query_type": None, "documents": [], "generation": ""}

        result = await self.graph.ainvoke(initial_state, config=config)
        return result["generation"]
