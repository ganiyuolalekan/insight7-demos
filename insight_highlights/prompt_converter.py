# Utility Functions

import re
import time

import streamlit as st

from func_store import group_insights_by_similarity, group_similar_sentences, time_calculator

from gpt_calls import get_answer, extract_insights

from func_store import align_docs
from prompts import survey_prompt, survey_query
from utils import embed_docs, search_docs, text_to_docs, text_to_docs_survey

from prompts import i_user_prompt, query, user_prompt_2, query_for_tags, user_prompt_1


def process_response(response, tags):

    match = re.search(r"{(.|\n)*}", response)
    if match:
        result = eval(match.group())
        topics = list(result.keys())
        categories = [' '.join([_tag.capitalize() for _tag in tag.split(' ')]) for tag in tags]

        for topic in topics:
            sentiment_count = sum([
                len(result[topic][cat])
                for cat in categories
            ])

            result[topic]['Sentiment Count'] = sentiment_count

        return {'data': result}


def get_result(doc, tags):
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

    return source_ref, process_response(response, tags)


def get_prompted_result(doc, tags):
    # query = query_for_tags(tags)
    docs = text_to_docs(doc)
    index = embed_docs(docs)
    time.sleep(2)
    sources = search_docs(index, query_for_tags(tags))
    source_ref = {
        source.metadata['source']: source.page_content
        for source in sources
    }
    prompt = user_prompt_1 + user_prompt_2(source_ref)
    text = get_answer(sources, query)['output_text']
    response = extract_insights(text, prompt)

    return source_ref, process_response(response, tags)


def get_prompted_survey_result(docs):
    groups = group_similar_sentences(docs)
    docs = text_to_docs_survey(['\n\n'.join(group) for group in groups if len(group) > 1])
    index = embed_docs(docs)
    sources = search_docs(index, survey_query)
    source_ref = {
        source.metadata['source']: source.page_content
        for source in sources
    }
    text = get_answer(sources, survey_query)['output_text']
    response = eval(extract_insights(text, survey_prompt))

    result = {}
    for tag in response.keys():
        result[tag] = []
        for insight in response[tag]:
            topic = insight['topic']
            highlights = source_ref[insight['highlights']].split('\n\n')
            count = len(highlights)

            result[tag].append({
                'topic': topic,
                'highlights': highlights,
                'count': count
            })

    return result


def use_prompt_gen(doc, q, p, tags):
    docs = text_to_docs(doc)
    index = embed_docs(docs)
    time.sleep(2)
    sources = search_docs(index, q)
    source_ref = {
        source.metadata['source']: source.page_content
        for source in sources
    }
    prompt = p + user_prompt_2(source_ref)
    text = get_answer(sources, query)['output_text']
    response = extract_insights(text, prompt)

    return source_ref, process_response(response, tags)


def generate_survey_result(results, tags, threshold=.4):
    res = {tag: [] for tag in tags}

    for tag in tags:
        insights = results[tag.lower()[:-1]]
        groups = group_insights_by_similarity(insights, threshold=threshold)
        if groups is not None:
            for group in groups:
                res[tag].append({
                    'topic': group,
                    'highlights': groups[group],
                    'count': len(groups[group])
                })

    return res


def get_result_frequency(data, thresh=.35):
    start_time = time.time()
    tracker = align_docs(data)
    tracker_time = time.time()
    time_calculator(
        "Document sorting time - responsible for processing csv and other extensions",
        start_time, tracker_time)
    groups = group_similar_sentences(list(tracker.keys()), threshold=thresh)
    grouping_time = time.time()
    time_calculator(
        "Similar text grouping time - responsible for grouping similar sentences using the specified threshold",
        tracker_time, grouping_time)
    docs = text_to_docs_survey(['\n\n'.join(group) for group in groups if len(group) > 1])
    index = embed_docs(docs)
    embedding_time = time.time()
    time_calculator(
        "Embedding time - time taken to embed the document",
        grouping_time, embedding_time)
    sources = search_docs(index, query)
    source_ref = {
        str(source.metadata['source']): source.page_content
        for source in sources
    }
    text = get_answer(sources, query)['output_text']
    first_prompt_layer_time = time.time()
    time_calculator(
        "Time to query document using prompt",
        embedding_time, first_prompt_layer_time)
    response = eval(extract_insights(text, survey_prompt))
    second_prompt_layer_time = time.time()
    time_calculator(
        "Time to compute final result using prompt",
        first_prompt_layer_time, second_prompt_layer_time)
    time_calculator(
        "The program (API) executed for",
        start_time, second_prompt_layer_time)

    result = {}
    for tag in response.keys():
        result[tag] = []
        for insight in response[tag]:
            topic = insight['topic']
            highlights = [
                [text, tracker[text]]
                for text in source_ref[insight['highlights']].split('\n\n')
            ]
            count = len(highlights)

            result[tag].append({
                'topic': topic,
                'highlights': highlights,
                'count': count
            })

    return {'data': result}

