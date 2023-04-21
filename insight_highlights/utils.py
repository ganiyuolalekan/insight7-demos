# Utility Functions

# from llms import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

from typing import Any, Dict, List

from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# DOCUMENT STORES FUNCTIONS
def text_to_docs(text) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


def text_to_docs_survey(texts) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""

    page_docs = [Document(page_content=page) for page in texts]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1
        doc.metadata["source"] = i + 1

    return page_docs


# ------------------------------------------------------------------------------------------------
# EMBEDDING FUNCTIONS
def embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    # embeddings = OpenAIEmbeddings(
    #     # openai_api_key=st.session_state.get("OPENAI_API_KEY")
    # )  # type: ignore
    embeddings = OpenAIEmbeddings(
        # deployment="sample_deployment",
        model="text-embedding-ada-002",
        chunk_size=5000
    )
    index = FAISS.from_documents(docs, embeddings)

    return index


# ------------------------------------------------------------------------------------------------
# SEARCH DOCUMENT FUNCTIONS
def search_docs(index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""

    # Search for similar chunks
    docs = index.similarity_search(query, k=20)
    return docs


def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs
