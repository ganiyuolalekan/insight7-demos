# Utility Functions

import os
import re
import time
import openai
import docx2txt

import streamlit as st

from prompts import query, i_user_prompt, STUFF_PROMPT
from io import BytesIO
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator
from sentence_transformers import SentenceTransformer, util
from openai.error import APIConnectionError, APIError, RateLimitError, Timeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from langchain.llms import OpenAI
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


THRESH = .2
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def app_meta():
    """Adds app meta data to web applications"""

    # Set website details
    st.set_page_config(
        page_title="Insight7 | Frequency based themes",
        page_icon="images/insight.png",
        layout='centered'
    )

def divider():
    """Sub-routine to create a divider for webpage contents"""

    st.markdown("""---""")


def get_sentiment(sentiment_name, data):
    """Gets the sentiments data by the sentiment name"""
    
    result = {}
    documents = data['data'].keys()

    for doc in documents:
        sentiment_result = {}
        for insights in list(data['data'][doc].values()):
            for (insight, highlight) in insights[sentiment_name]:
                sentiment_result[insight] = highlight
        result[doc] = sentiment_result

    return result


def compute_similarity(text_1, text_2):
    """computes similarity between two texts"""

    embedding_1 = similarity_model.encode(text_1, convert_to_tensor=True)
    embedding_2 = similarity_model.encode(text_2, convert_to_tensor=True)

    return round(float(util.pytorch_cos_sim(embedding_1, embedding_2)[0][0]), 4)


def find_similar_insights(*lists, thresh=THRESH):
    """Finds all similar insights within the documents"""

    output = []

    for i, lst in enumerate(lists):
        for j, sentence in enumerate(lst):
            similar_sentences = [[sentence, i]]
            other_lists = [l for l in lists if l != lst]
            for k, other_list in enumerate(other_lists):
                for l, other_sentence in enumerate(other_list):
                    similarity_score = compute_similarity(sentence, other_sentence)
                    if similarity_score > thresh:
                        similar_sentences.append([other_sentence, lists.index(other_list)])

            if len(set([s[0] for s in similar_sentences])) == len(similar_sentences):
                is_unique = True
                for out_list in output:
                    if set([s[0] for s in similar_sentences]).intersection([s[0] for s in out_list]):
                        is_unique = False
                        break
                if is_unique and len(similar_sentences) > 1:
                    output.append(similar_sentences)

    return output


def pick_similar_insights(similar_insights):
    """Picks a topic for all similar insights"""

    _similar_insights = [
        [item[0] for item in lst]
        for lst in similar_insights
    ]
    insights_header = []

    for insights in _similar_insights:
        if len(insights_header):
            for insight in insights_header:
                _insights = insights.copy()
                _insights_count = len(_insights)
                if (insight in _insights) and (_insights_count > 2):
                    insights.remove(insight)
                    _insights_count -= 1

        word_count = list(map(lambda s: len(s.split(' ')), insights))
        insights_header.append(insights[word_count.index(max(word_count))])

    result = {
        insight_header: similar_insight
        for insight_header, similar_insight in zip(
            insights_header, similar_insights
        )
    }

    return result


def insight_highlight_mapper(sentiment_data, insights_topic):
    """Maps insight to highlights"""

    document_names = list(sentiment_data.keys())

    insight_highlight = {}
    for doc in sentiment_data.keys():
        insight_highlight.update(sentiment_data[doc])

    highlight_count = 0
    for it in insights_topic:
        for i, insight in enumerate(insights_topic[it]):
            insights_topic[it][i][0] = insight_highlight[insight[0]]
            insights_topic[it][i][1] = document_names[insight[1]]
            highlight_count += 1

    insights_topic["Highlight Count"] = highlight_count

    return insights_topic


def generate_insights_highlights(data, tags, thresh):
    result = {'data': {}}

    for tag in tags:
        tag_sentiments = get_sentiment(tag, data)

        similar_insights = find_similar_insights(*[
            list(l.keys())
            for l in tag_sentiments.values()
        ], thresh=thresh)

        insight_topic_map = pick_similar_insights(similar_insights)

        tag_outcome = insight_highlight_mapper(tag_sentiments, insight_topic_map)

        result['data'][tag] = tag_outcome

    return result

class OpenAIEmbeddings(BaseModel, Embeddings):
    """Wrapper around OpenAI embedding models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import OpenAIEmbeddings
            openai = OpenAIEmbeddings(openai_api_key="my-api-key")
    """

    client: Any  #: :meta private:
    document_model_name: str = "text-embedding-ada-002"
    query_model_name: str = "text-embedding-ada-002"
    openai_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    # TODO: deprecate this
    @root_validator(pre=True, allow_reuse=True)
    def get_model_names(cls, values: Dict) -> Dict:
        """Get model names from just old model name."""
        if "model_name" in values:
            if "document_model_name" in values:
                raise ValueError(
                    "Both `model_name` and `document_model_name` were provided, "
                    "but only one should be."
                )
            if "query_model_name" in values:
                raise ValueError(
                    "Both `model_name` and `query_model_name` were provided, "
                    "but only one should be."
                )
            model_name = values.pop("model_name")
            values["document_model_name"] = f"text-search-{model_name}-doc-001"
            values["query_model_name"] = f"text-search-{model_name}-query-001"
        return values

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
            values["client"] = openai.Embedding
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=10, max=60),
        retry=(
                retry_if_exception_type(Timeout)
                | retry_if_exception_type(APIError)
                | retry_if_exception_type(APIConnectionError)
                | retry_if_exception_type(RateLimitError)
        ),
    )
    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint with exponential backoff."""
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return self.client.create(input=[text], engine=engine)["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        responses = [
            self._embedding_func(text, engine=self.document_model_name)
            for text in texts
        ]
        return responses

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embedding_func(text, engine=self.query_model_name)
        return embedding


def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


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


def embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    embeddings = OpenAIEmbeddings(
        # openai_api_key=st.session_state.get("OPENAI_API_KEY")
    )  # type: ignore
    index = FAISS.from_documents(docs, embeddings)

    return index


def search_docs(index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""

    # Search for similar chunks
    docs = index.similarity_search(query, k=7)
    return docs


def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    chain = load_qa_with_sources_chain(
        OpenAI(
            temperature=0.01,
            model_name="text-davinci-003",
            max_tokens=1000
        ),
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )

    answer = chain(
        {
            "input_documents": docs,
            "question": query
        }, return_only_outputs=True
    )

    return answer


def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


def wrap_text_in_html(text) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


def extract_insights(text, user_prompt):
    openai.api_key = OPENAI_API_KEY
    conversation = [{'role': 'system', 'content': f'{user_prompt}:{text}'}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation,
        temperature=0.01,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0]['message']['content']


def clean_doc(text, user_prompt):
    openai.api_key = OPENAI_API_KEY
    conversation = [{'role': 'system', 'content': f'{user_prompt}:{text}'}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation,
        temperature=0.01,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0]['message']['content']


def process_response(response, sources):

    match = re.search(r"{(.|\n)*}", response)
    if match:
        result = eval(match.group())
        topics = list(result.keys())
        categories = ["Pain Points", "Desires", "Behaviour"]

        for topic in topics:
            sentiment_count = sum([
                len(result[topic][cat])
                for cat in categories
            ])

            result[topic]['Sentiment Count'] = sentiment_count

        return {'data': result}


def get_result(doc):
    docs = text_to_docs(doc)
    index = embed_docs(docs)
    time.sleep(2)
    sources = search_docs(index, query)
    source_ref = {
        source.metadata['source']: source.page_content
        for source in sources
    }
    answer = get_answer(sources, query)
    time.sleep(2)
    text = answer['output_text']
    response = extract_insights(text, i_user_prompt(source_ref))

    return process_response(response, sources)


def get_prompted_result(doc, prompt):
    docs = text_to_docs(doc)
    index = embed_docs(docs)
    time.sleep(2)
    sources = search_docs(index, query)
    source_ref = {
        source.metadata['source']: source.page_content
        for source in sources
    }
    text = get_answer(sources, query)['output_text']
    response = extract_insights(text, prompt)

    return source_ref, process_response(response, sources)
