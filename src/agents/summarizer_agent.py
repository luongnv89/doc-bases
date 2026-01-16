"""
Summarizer agent for condensing documents.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.utils.logger import get_logger

logger = get_logger()


class SummarizerAgent:
    """Agent specialized in summarization."""

    def __init__(self, llm):
        self.llm = llm

        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Create concise summaries focused on answering questions.
Extract the most relevant information and present it clearly.""",
                ),
                (
                    "human",
                    """Summarize the following documents to help answer this question:

Question: {question}

Documents:
{documents}

Provide a focused summary:""",
                ),
            ]
        )

        logger.info("SummarizerAgent initialized")

    async def summarize(self, documents: list[Document], question: str) -> str:
        """Generate query-focused summary."""
        doc_text = "\n\n---\n\n".join([d.page_content for d in documents])

        chain = self.summary_prompt | self.llm
        response = await chain.ainvoke({"question": question, "documents": doc_text})

        result = response.content if hasattr(response, "content") else str(response)
        logger.info(f"Generated summary of {len(result)} chars from {len(documents)} docs")
        return result

    def summarize_sync(self, documents: list[Document], question: str) -> str:
        """Generate query-focused summary synchronously."""
        doc_text = "\n\n---\n\n".join([d.page_content for d in documents])

        chain = self.summary_prompt | self.llm
        response = chain.invoke({"question": question, "documents": doc_text})

        result = response.content if hasattr(response, "content") else str(response)
        logger.info(f"Generated summary of {len(result)} chars from {len(documents)} docs")
        return result

    async def map_reduce_summarize(self, documents: list[Document], question: str) -> str:
        """Map-reduce summarization for large document sets."""
        if len(documents) <= 3:
            # Small enough to summarize directly
            return await self.summarize(documents, question)

        # Map: Summarize each document individually
        summaries = []
        for doc in documents:
            summary = await self.summarize([doc], question)
            summaries.append(summary)

        # Reduce: Combine summaries into final summary
        combined = Document(page_content="\n\n".join(summaries))
        final = await self.summarize([combined], question)

        logger.info(f"Map-reduce: {len(documents)} docs -> {len(summaries)} summaries -> final")
        return final

    def extract_key_points(self, documents: list[Document], question: str, max_points: int = 5) -> list[str]:
        """Extract key points from documents."""
        doc_text = "\n\n".join([d.page_content for d in documents])

        prompt = f"""Extract up to {max_points} key points from these documents that help answer:
Question: {question}

Documents:
{doc_text}

Return as a numbered list of key points:"""

        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Parse numbered list
        lines = content.strip().split("\n")
        points = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove number/bullet prefix
                point = line.lstrip("0123456789.-) ").strip()
                if point:
                    points.append(point)

        return points[:max_points]
