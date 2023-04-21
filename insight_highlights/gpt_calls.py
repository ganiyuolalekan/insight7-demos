import os
import openai

from typing import Any, Dict, List
from prompts import STUFF_PROMPT

from langchain.llms import OpenAIChat
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    openai.api_key = OPENAI_API_KEY

    chain = load_qa_with_sources_chain(
        OpenAIChat(
            temperature=0.01,
            model_name="gpt-4",
            max_tokens=3000
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


def extract_insights(text, user_prompt):
    openai.api_key = OPENAI_API_KEY
    conversation = [{'role': 'system', 'content': f'{user_prompt}:{text}'}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation,
        temperature=0.01,
        max_tokens=4000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0]['message']['content']


def extract_insights_clean(text, user_prompt):
    openai.api_key = OPENAI_API_KEY
    conversation = [{'role': 'system', 'content': f'{user_prompt}:{text}'}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation,
        temperature=0.01,
        max_tokens=1500,
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
