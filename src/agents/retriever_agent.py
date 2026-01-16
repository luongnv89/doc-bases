"""
Specialized retriever agent with query decomposition.
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.utils.logger import get_logger

logger = get_logger()


class RetrieverAgent:
    """Agent specialized in document retrieval."""

    def __init__(self, vectorstore, llm, checkpointer=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.checkpointer = checkpointer
        self.agent = self._create_agent()
        logger.info("RetrieverAgent initialized")

    def _create_agent(self):
        """Create ReAct agent with retrieval tools."""
        vectorstore = self.vectorstore
        llm = self.llm

        @tool
        def retrieve_documents(query: str, k: int = 4) -> str:
            """Retrieve relevant documents from knowledge base."""
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)
            return "\n\n".join([d.page_content for d in docs])

        @tool
        def decompose_query(complex_query: str) -> str:
            """Break complex query into simpler sub-queries."""
            prompt = f"""Break this question into 2-3 simpler queries:
Question: {complex_query}

Return as numbered list."""
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)

        tools = [retrieve_documents, decompose_query]
        return create_react_agent(self.llm, tools, checkpointer=self.checkpointer)

    async def retrieve(self, question: str, config: dict = None) -> dict:
        """Execute retrieval workflow."""
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": f"Retrieve relevant information for: {question}"}]},
            config=config
        )
        return result

    def retrieve_sync(self, question: str, config: dict = None) -> dict:
        """Execute retrieval workflow synchronously."""
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": f"Retrieve relevant information for: {question}"}]},
            config=config
        )
        return result

    def get_documents(self, question: str, k: int = 5) -> List[Document]:
        """Direct document retrieval without agent."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(question)
