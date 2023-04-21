import re
import nltk
import docx2txt
import numpy as np
import streamlit as st

from io import BytesIO
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from typing import Dict, List

nltk.download('punkt')
nltk.download('stopwords')

THRESH = .35
MIN_SENTENCES = 2
TOP_PERCENTAGE = .7
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def sum_list(list_item):
    result = list_item[0]
    for item in list_item[1:]:
        result += item
    return result


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

def wrap_text_in_html(text) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


def similarity(s1, s2):
    """Calculate the similarity between two strings"""
    return SequenceMatcher(None, s1, s2).ratio()


def sort_by_similarity(strings):
    """Sort a list of strings by their ascending order of similarity"""
    return sorted(strings, key=lambda s: sum(similarity(s, t) for t in strings))


def is_significant_text(text):
    # Define the threshold for significant text
    threshold = 80  # You can adjust this value as per your requirement

    # Count the number of alphanumeric characters in the text
    alphanumeric_count = sum(1 for char in text if char.isalnum())

    # Check if the text contains a significant amount of alphanumeric characters
    return alphanumeric_count >= threshold


def group_by_sentiment(outcomes):
    group = {
        'pain point': [],
        'desire': [],
        'behaviour': []
    }

    for outcome in outcomes:
        _class = outcome[2]
        outcome.remove(_class)
        outcome.remove(outcome[0])
        group[_class].append(outcome)

    return group


def group_insights_by_similarity(insights, threshold=0.4):
    grouped_insights = {}
    for insight, text in insights:
        insights = [insight for insight in insights if len(insight) == 2]
        # Check if any existing insight is similar to the current insight
        is_similar = False
        for existing_insight in grouped_insights.keys():
            _similarity = SequenceMatcher(None, insight, existing_insight).ratio()
            if _similarity >= threshold:
                # Group the current insight with the most similar one
                grouped_insights[existing_insight].append(text)
                is_similar = True
                break

        if not is_similar:
            # Add the current insight as a new group
            grouped_insights[insight] = [text]

    # Sort the groups based on the length of the insight (longest to shortest)
    sorted_groups = sorted(grouped_insights.items(), key=lambda x: len(x[0]), reverse=True)

    # Convert the sorted groups to a dictionary and return it
    return {insight: texts for insight, texts in sorted_groups}


def group_similar_sentences(reviews, threshold=THRESH):
    # Vectorize reviews using TfidfVectorize
    vectorize = TfidfVectorizer(stop_words='english')
    x = vectorize.fit_transform(reviews)

    # Compute pairwise cosine similarity
    pairwise_similarity = np.dot(x, x.T)

    # Group reviews into clusters
    n_reviews = len(reviews)
    visited = set()
    clusters = []
    for i in range(n_reviews):
        if i not in visited:
            cluster = []
            cluster.append(reviews[i])
            visited.add(i)
            for j in range(i + 1, n_reviews):
                if j not in visited and pairwise_similarity[i, j] >= threshold:
                    cluster.append(reviews[j])
                    visited.add(j)
            clusters.append(cluster)

    return clusters


def group_similar_sentences_v2(reviews, threshold=0.5):
    # Vectorize reviews using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(reviews)

    # Compute pairwise cosine similarity
    similarity_matrix = csr_matrix(x @ x.T)
    connected_components_matrix = connected_components(similarity_matrix > threshold)

    # Group reviews into clusters
    clusters = [[] for _ in range(connected_components_matrix[0])]
    for i, cluster_id in enumerate(connected_components_matrix[1]):
        clusters[cluster_id].append(reviews[i])

    return clusters[1:]


def group_similar_sentences_v3(reviews, threshold=THRESH):
    # Vectorize reviews using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(reviews)

    # Compute pairwise cosine similarity
    pairwise_similarity = np.dot(x, x.T)

    # Group reviews into clusters
    n_reviews = len(reviews)
    visited = set()
    clusters = []
    for i in range(n_reviews):
        if i not in visited:
            cluster = []
            cluster.append(reviews[i])
            visited.add(i)
            for j in range(i + 1, n_reviews):
                if j not in visited and pairwise_similarity[i, j] >= threshold:
                    cluster.append(reviews[j])
                    visited.add(j)
            clusters.append(cluster)

    return clusters



def preprocess_text(text, min_sentences=MIN_SENTENCES):
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    # Group sentences into relevant texts consisting of at least min_sentences sentences
    relevant_texts = []
    current_text = []
    for sentence in sentences:
        current_text.append(sentence)
        if len(current_text) >= min_sentences:
            relevant_texts.append(current_text.copy())
            current_text.clear()

    # Append any remaining sentences to the last relevant text
    if len(current_text) > 0:
        relevant_texts[-1].extend(current_text)

    # Convert relevant texts back to single strings
    relevant_text_strings = []
    for text in relevant_texts:
        relevant_text_strings.append(" ".join(text))

    return relevant_text_strings

def preprocess_text_v2(text, min_sentences=5):
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    # Convert sentences to a NumPy array for faster processing
    sentences_array = np.array(sentences)

    # Compute the number of relevant texts
    num_relevant_texts = len(sentences) // min_sentences

    # Compute the indices of the relevant texts
    relevant_text_indices = np.arange(num_relevant_texts) * min_sentences

    # Split the sentences into relevant texts
    relevant_texts = np.split(sentences_array, relevant_text_indices[1:])

    # Convert relevant texts back to single strings
    relevant_text_strings = [" ".join(text) for text in relevant_texts]

    return relevant_text_strings



def vectorize_sentences(sentences):
    vectorize = TfidfVectorizer()
    sentence_vectors = vectorize.fit_transform(sentences)

    return sentence_vectors


def rank_sentences(sentence_vectors):
    sentence_scores = cosine_similarity(sentence_vectors)
    return sentence_scores


def summarize_text(text, thresh=TOP_PERCENTAGE):
    relevant_sentences = preprocess_text_v2(text)
    sentence_vectors = vectorize_sentences(relevant_sentences)
    sentence_scores = rank_sentences(sentence_vectors)

    ranked_sentences = [(score, sentence) for score, sentence in zip(sentence_scores.tolist()[0], relevant_sentences)]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[0], reverse=True)

    all_sentences = [sentence[1] for sentence in ranked_sentences]
    top_sentences = [sentence[1] for sentence in ranked_sentences[:int(len(all_sentences) * thresh)]]

    return top_sentences


def align_docs(data):
    tracker = {}

    groups = {
        'csv': {},
        'others': {}
    }

    for _id, _data in zip(
            list(data.keys()),
            list(data.values())
    ):
        groups['csv' if _data[1].lower() == 'csv' else 'others'][str(_id)] = _data[0]

    if len(groups['others']):
        for _id, doc in zip(
                list(groups['others'].keys()),
                list(groups['others'].values())
        ):
            texts = summarize_text(doc)
            for text in texts:
                tracker[text] = _id

    if len(groups['csv']):
        for _id, docs in zip(
                list(groups['csv'].keys()),
                list(groups['csv'].values())
        ):
            docs = docs.split('\n\n')
            for doc in docs:
                tracker[doc] = _id

    return tracker


def align_docs_v2(data: Dict[str, tuple]) -> Dict[str, str]:
    tracker = {}
    groups = defaultdict(list)

    for _id, (_data, format_type) in data.items():
        groups[format_type.lower()].append((_id, _data))

    if groups['others']:
        for _id, doc in groups['others']:
            texts = summarize_text(doc)
            for text in texts:
                tracker[text] = _id

    if groups['csv']:
        for _id, docs in groups['csv']:
            docs = docs.split('\n\n')
            for doc in docs:
                tracker[doc] = _id

    return tracker



def time_calculator(process_name, start_time, end_time):
    process_time = round((end_time - start_time), 2)
    st.markdown(f"> **{process_name}**: took `{process_time}` secs to execute!\n")
