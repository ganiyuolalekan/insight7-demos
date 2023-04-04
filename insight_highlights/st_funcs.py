# Test for how the cleaning layer should function

import pickle
import random

import streamlit as st

from api_call import file_extractor
from read_files import read_uploaded_file
from documentation import document_correction, prompt_testing
from prompts import clean_document_prompt, query, user_prompt_1
from utils import clean_doc, divider, get_prompted_result, generate_insights_highlights, use_prompt_gen


def test_doc(accept_multiple_files=False, return_used_test=False):
    use_test_output = st.checkbox(
        label="Use test document",
        help="Quickly evaluate functions",
        value=False
    )
    st.write("> Check this to quickly test a preloaded document")
    divider()

    if use_test_output:
        # Production
        files = [
            "insight_highlights/data/doc1.txt",
            "insight_highlights/data/doc2.txt",
            "insight_highlights/data/doc3.txt",
            "insight_highlights/data/doc4.txt"
        ]

        # Local
        # files = [
        #     "data/doc1.txt",
        #     "data/doc2.txt",
        #     "data/doc3.txt",
        #     "data/doc4.txt"
        # ]

        doc_file = random.choice(files)

        if accept_multiple_files:
            document = {
                doc.split('/')[-1]: open(doc, 'r').read()
                for doc in files
            }
        else:
            document = open(doc_file, 'r').read()
        st.success("Using a preloaded test document", icon='✅')
    else:

        if not accept_multiple_files:
            use_url = st.checkbox(
                label="Make use of a file url to test",
                help="Uses API to load content from url",
                value=False
            )
        else:
            use_url = False

        if use_url and not accept_multiple_files:
            file_url = st.text_input("File URL:")
            if not len(file_url) > 0:
                document = None
                st.warning("File URL file is empty! Please enter a URL", icon="⚠️")
            else:
                document = file_extractor(file_url)
                st.success("File extracted successfully", icon='✅')
        else:
            document = st.file_uploader("Enter a document file:", accept_multiple_files=accept_multiple_files)

            if accept_multiple_files:
                document = {
                    doc.name: read_uploaded_file(doc)
                    for doc in document
                }
            else:
                if document is not None:
                    document = read_uploaded_file(document)
            st.success("Successfully read document", icon='✅')
    divider()

    if return_used_test:
        return document, use_test_output
    else:
        return document


def doc_correction():
    """Document Fine-Tuning"""

    st.markdown(document_correction, unsafe_allow_html=True)
    divider()

    document = test_doc()

    if document is not None:
        cleaned_document = clean_doc(document, clean_document_prompt)
        input_doc, modified_doc = st.tabs(["Uploaded document", "Cleaned Document"])

        with input_doc:
            st.header("Here's the uploaded document")
            st.write(document)

        with modified_doc:
            st.header("Here's cleaned document")
            st.write(cleaned_document)


def prompt_eng():
    """Prompt engineering"""

    st.markdown(prompt_testing, unsafe_allow_html=True)
    divider()

    document = test_doc()

    user_query = st.text_area(
        "Enter your query:",
        value=query, height=400,
        help="Below is the current prompt being used"
    )

    user_prompt = st.text_area(
        "Enter your prompt:",
        value=user_prompt_1, height=600,
        help="Below is the current prompt being used"
    )

    analyze = st.button("Analyze document using prompt")

    if analyze:
        if document is not None:
            source_ref, response = use_prompt_gen(
                document, user_query, user_prompt,
                ['Pain Points', 'Desires', 'Behaviours']
            )
            st.success("Successfully analyzed document", icon='✅')

            input_doc, source_references, api_result = st.tabs([
                "Uploaded document",
                "Source reference from document",
                "Result from API"
            ])

            with input_doc:
                st.header("Here's the uploaded document")
                st.write(document)

            with source_references:
                st.header("Reference text with insights from document")
                st.write(source_ref)

            with api_result:
                st.header("Here's the result from your prompt")
                st.write(response)

            prompt_name = st.text_input("Enter prompt title:", value='Prompt-1')
            save_prompt = st.button("Save the prompt used for this result")

            if save_prompt:
                st.warning("Note: if you have modified the prompt after analyse... the modified result will be saved", icon="⚠️")
                with open(prompt_name, 'w') as f:
                    print(user_prompt, file=f)
        else:
            st.warning("Please upload a document or use the preloaded document", icon="⚠️")


def theme_api():
    # st.markdown(prompt_testing, unsafe_allow_html=True)
    # divider()

    document = test_doc()

    specified_tags = st.multiselect(
        'Select the tags to apply to the API',
        ['Pain Points', 'Desires', 'Behaviours', 'Bugs', 'Threads']
    )

    # specified_tags = ['Pain Points', 'Desires', 'Behaviours']

    analyze = st.button("Analyze document")

    if analyze:
        if document is not None:
            source_ref, response = get_prompted_result(document, specified_tags)
            st.success("Successfully analyzed document", icon='✅')

            input_doc, source_references, api_result = st.tabs([
                "Uploaded document",
                "Source reference from document",
                "Result from API"
            ])

            with input_doc:
                st.header("Here's the uploaded document")
                st.write(document)

            with source_references:
                st.header("Reference text with insights from document")
                st.write(source_ref)

            with api_result:
                st.header("Here's the result from your prompt")
                st.write(response)
        else:
            st.warning("Please upload a document or use the preloaded document", icon="⚠️")


def frequency_api():
    documents, use_test_output = test_doc(accept_multiple_files=True, return_used_test=True)

    if len(documents):
        preview_doc = st.selectbox(
            label="Choose a document to preview",
            options=list(documents.keys())
        )

        expander = st.expander("See document")
        expander.write(documents[preview_doc])
        divider()

        theme_similarity_thresh = st.slider(
            "Similarity Thresh", min_value=.0, max_value=1.00, value=.20, step=.001,
            help="The greater the theme similarity, the more strict the insights picked"
        )

        specified_tags = st.multiselect(
            'Select the tags to apply to the API',
            ['Pain Points', 'Desires', 'Behaviours', 'Bugs', 'Threads']
        )

        # specified_tags = ['Pain Points', 'Desires', 'Behaviours']

        analyze = st.button("Analyze documents")

        if analyze:
            if use_test_output:
                data = pickle.load(open('data/serialized_theme.p', 'rb'))
                insights_results = pickle.load(open('data/serialized_frequency.p', 'rb'))
                # insights_results = generate_insights_highlights(
                #     data=data,
                #     tags=specified_tags,
                #     thresh=theme_similarity_thresh
                # )
                # pickle.dump(insights_results, open('data/serialized_frequency.p', 'wb'))
                st.success("Successfully analyzed documents and retrieved insights", icon='✅')
                divider()
                st.success("Successfully mapped similar highlights to corresponding insights", icon='✅')
                divider()
            else:
                data = {'data': {
                    doc: get_prompted_result(doc, specified_tags)[1]['data']
                    for doc in documents
                }}
                st.success("Successfully analyzed documents and retrieved insights", icon='✅')
                divider()

                insights_results = generate_insights_highlights(
                    data=data,
                    tags=specified_tags,
                    thresh=theme_similarity_thresh
                )
                st.success("Successfully mapped similar highlights to corresponding insights", icon='✅')
                divider()

            st.write(insights_results)
