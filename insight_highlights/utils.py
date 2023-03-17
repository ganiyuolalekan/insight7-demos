# Utility Functions

import time
import requests

import streamlit as st

from sentence_transformers import SentenceTransformer, util


THRESH = .2
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


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


def generate_insight_api(data, count=0):
    """Generates the result"""

    time.sleep(2)

    result = requests.request(
        "POST", "https://insight7-aa-azi5xvdx6a-uc.a.run.app/insights",
        data={'input_text': data}
    ).text

    try:
        eval(result)

        return result
    except NameError or SyntaxError:
        print("Retried")
        count += 1
        if count < 5:
            generate_insight_api(data, count)

        return ""
