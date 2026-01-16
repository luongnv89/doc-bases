"""
RAG quality evaluation using LLM-based grading.
Provides relevance scoring and hallucination detection.
"""
from typing import List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.utils.logger import get_logger

logger = get_logger()


class RelevanceScore(BaseModel):
    """Structured output for relevance grading."""
    score: float = Field(description="Relevance score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation for the score")


class HallucinationCheck(BaseModel):
    """Structured output for hallucination detection."""
    is_grounded: bool = Field(description="Whether answer is grounded in documents")
    unsupported_claims: List[str] = Field(
        default=[],
        description="Claims not supported by documents"
    )


class RAGEvaluator:
    """
    Evaluate RAG retrieval and generation quality.

    Used by Corrective RAG to decide if retrieval is sufficient.
    """

    def __init__(self, llm):
        self.llm = llm
        self.relevance_parser = PydanticOutputParser(pydantic_object=RelevanceScore)
        self.hallucination_parser = PydanticOutputParser(pydantic_object=HallucinationCheck)

        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a relevance grader. Evaluate if a document is relevant to answering a question.

Score from 0.0 (completely irrelevant) to 1.0 (highly relevant).

{format_instructions}"""),
            ("human", """Question: {question}

Document: {document}

Grade the relevance:""")
        ])

        self.hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker. Determine if an answer is fully grounded in the provided documents.

{format_instructions}"""),
            ("human", """Documents:
{documents}

Answer to check:
{answer}

Is this answer grounded in the documents?""")
        ])

    async def grade_relevance(self, question: str, document: str) -> RelevanceScore:
        """Grade relevance of a document to a question."""
        chain = self.relevance_prompt | self.llm | self.relevance_parser
        result = await chain.ainvoke({
            "question": question,
            "document": document,
            "format_instructions": self.relevance_parser.get_format_instructions()
        })
        logger.debug(f"Relevance score: {result.score}")
        return result

    async def check_hallucination(self, documents: str, answer: str) -> HallucinationCheck:
        """Check if answer contains hallucinations."""
        chain = self.hallucination_prompt | self.llm | self.hallucination_parser
        result = await chain.ainvoke({
            "documents": documents,
            "answer": answer,
            "format_instructions": self.hallucination_parser.get_format_instructions()
        })
        logger.debug(f"Grounded: {result.is_grounded}")
        return result

    async def grade_documents_batch(
        self,
        question: str,
        documents: List[str],
        threshold: float = 0.5
    ) -> Tuple[List[str], List[str]]:
        """
        Grade multiple documents and split into relevant/irrelevant.

        Returns:
            Tuple of (relevant_docs, irrelevant_docs)
        """
        relevant = []
        irrelevant = []

        for doc in documents:
            score = await self.grade_relevance(question, doc)
            if score.score >= threshold:
                relevant.append(doc)
            else:
                irrelevant.append(doc)

        logger.info(f"Graded {len(documents)} docs: {len(relevant)} relevant")
        return relevant, irrelevant
