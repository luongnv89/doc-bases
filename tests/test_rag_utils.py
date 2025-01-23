# tests/test_rag_utils.py
from langchain.schema import Document
from src.utils.rag_utils import setup_rag, list_knowledge_bases


def test_setup_rag():
    # Mock documents using LangChain's Document class
    documents = [
        Document(page_content="This is a test document.", metadata={"source": "test"}),
        Document(page_content="Another test document.", metadata={"source": "test"}),
    ]
    knowledge_base_name = "test_kb"
    qa_chain = setup_rag(documents, knowledge_base_name)
    assert qa_chain is not None


def test_list_knowledge_bases():
    # Ensure the function returns a list
    knowledge_bases = list_knowledge_bases()
    assert isinstance(knowledge_bases, list)
