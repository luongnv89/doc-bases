"""
Critic agent for answer validation.
"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.utils.logger import get_logger

logger = get_logger()


class Critique(BaseModel):
    """Structured critique output."""
    accuracy_score: float = Field(description="Factual accuracy score from 0.0 to 1.0")
    completeness_score: float = Field(description="Completeness score from 0.0 to 1.0")
    clarity_score: float = Field(description="Clarity score from 0.0 to 1.0")
    issues: List[str] = Field(default=[], description="Issues found in the answer")
    suggestions: List[str] = Field(default=[], description="Suggestions for improvement")
    needs_revision: bool = Field(description="Whether the answer needs revision")


class CriticAgent:
    """Agent for answer validation and quality assessment."""

    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=Critique)

        self.critique_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critical evaluator. Assess answers for:
- Accuracy: Are claims supported by the documents?
- Completeness: Does it fully address the question?
- Clarity: Is it well-organized and easy to understand?

{format_instructions}"""),
            ("human", """Question: {question}

Source Documents:
{documents}

Answer to Evaluate:
{answer}

Provide your critique:""")
        ])

        logger.info("CriticAgent initialized")

    async def critique(self, question: str, answer: str, documents: str) -> Critique:
        """Critique an answer against source documents."""
        chain = self.critique_prompt | self.llm | self.parser

        try:
            result = await chain.ainvoke({
                "question": question,
                "documents": documents,
                "answer": answer,
                "format_instructions": self.parser.get_format_instructions()
            })

            logger.info(
                f"Critique scores: acc={result.accuracy_score:.2f}, "
                f"comp={result.completeness_score:.2f}, "
                f"clarity={result.clarity_score:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"Critique parsing failed: {e}")
            # Return a default critique on failure
            return Critique(
                accuracy_score=0.5,
                completeness_score=0.5,
                clarity_score=0.5,
                issues=["Could not parse critique"],
                suggestions=[],
                needs_revision=True
            )

    def critique_sync(self, question: str, answer: str, documents: str) -> Critique:
        """Critique an answer synchronously."""
        chain = self.critique_prompt | self.llm | self.parser

        try:
            result = chain.invoke({
                "question": question,
                "documents": documents,
                "answer": answer,
                "format_instructions": self.parser.get_format_instructions()
            })

            logger.info(
                f"Critique scores: acc={result.accuracy_score:.2f}, "
                f"comp={result.completeness_score:.2f}, "
                f"clarity={result.clarity_score:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"Critique parsing failed: {e}")
            return Critique(
                accuracy_score=0.5,
                completeness_score=0.5,
                clarity_score=0.5,
                issues=["Could not parse critique"],
                suggestions=[],
                needs_revision=True
            )

    async def suggest_improvements(self, answer: str, critique: Critique) -> str:
        """Improve answer based on critique."""
        if not critique.needs_revision:
            return answer

        issues_text = "\n".join([f"- {issue}" for issue in critique.issues])
        suggestions_text = "\n".join([f"- {s}" for s in critique.suggestions])

        prompt = f"""Improve this answer based on the feedback:

Original Answer:
{answer}

Issues Found:
{issues_text}

Suggestions:
{suggestions_text}

Provide an improved answer:"""

        response = await self.llm.ainvoke(prompt)
        improved = response.content if hasattr(response, 'content') else str(response)

        logger.info("Generated improved answer based on critique")
        return improved

    def suggest_improvements_sync(self, answer: str, critique: Critique) -> str:
        """Improve answer synchronously."""
        if not critique.needs_revision:
            return answer

        issues_text = "\n".join([f"- {issue}" for issue in critique.issues])
        suggestions_text = "\n".join([f"- {s}" for s in critique.suggestions])

        prompt = f"""Improve this answer based on the feedback:

Original Answer:
{answer}

Issues Found:
{issues_text}

Suggestions:
{suggestions_text}

Provide an improved answer:"""

        response = self.llm.invoke(prompt)
        improved = response.content if hasattr(response, 'content') else str(response)

        logger.info("Generated improved answer based on critique")
        return improved

    def get_overall_score(self, critique: Critique) -> float:
        """Calculate weighted overall score."""
        weights = {
            "accuracy": 0.5,
            "completeness": 0.3,
            "clarity": 0.2
        }

        score = (
            weights["accuracy"] * critique.accuracy_score +
            weights["completeness"] * critique.completeness_score +
            weights["clarity"] * critique.clarity_score
        )

        return round(score, 2)
