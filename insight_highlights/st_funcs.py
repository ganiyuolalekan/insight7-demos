# Test for how the cleaning layer should function

import random
from io import StringIO

import streamlit as st

from api_call import file_extractor
from utils import clean_doc, divider, get_prompted_result
from prompts import clean_document_prompt, user_prompt_1
from documentation import document_correction, prompt_testing


def doc_correction():
    """Document Fine-Tuning"""

    st.markdown(document_correction, unsafe_allow_html=True)
    divider()

    use_test_output = st.checkbox(
        label="Use test document",
        help="Quickly evaluate functions",
        value=False
    )
    st.write("> Check this to quickly test a preloaded document")
    divider()

    if use_test_output:
        doc_file = random.choice([
            "insight_highlights/data/doc1.txt",
            "insight_highlights/data/doc3.txt"
        ])
        document = open(doc_file, 'r').read()
        st.success("Using a preloaded test document", icon='✅')
    else:
        use_url = st.checkbox(
            label="Make use of a file url to test",
            help="Uses API to load content from url",
            value=False
        )

        if use_url:
            file_url = st.text_input("File URL:")
            if not len(file_url) > 0:
                document = None
                st.warning("File URL file is empty! Please enter a URL", icon="⚠️")
            else:
                document = file_extractor(file_url)
                st.success("File extracted successfully", icon='✅')
        else:
            document = st.file_uploader("Enter a document file:")
            if document is not None:
                document = StringIO(document.getvalue().decode("utf-8")).read()
                st.success("Successfully read document", icon='✅')
    divider()

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

    use_test_output = st.checkbox(
        label="Use test document",
        help="Quickly evaluate functions",
        value=False
    )
    st.write("> Check this to quickly test a preloaded document")
    divider()

    if use_test_output:
        doc_file = random.choice([
            "insight_highlights/data/doc1.txt",
            "insight_highlights/data/doc3.txt"
        ])
        document = open(doc_file, 'r').read()
        st.success("Using a preloaded test document", icon='✅')
    else:
        use_url = st.checkbox(
            label="Make use of a file url to test",
            help="Uses API to load content from url",
            value=False
        )

        if use_url:
            file_url = st.text_input("File URL:")
            if not len(file_url) > 0:
                document = None
                st.warning("File URL file is empty! Please enter a URL", icon="⚠️")
            else:
                document = file_extractor(file_url)
                st.success("File extracted successfully", icon='✅')
        else:
            document = st.file_uploader("Enter a document file:")
            if document is not None:
                document = StringIO(document.getvalue().decode("utf-8")).read()
                st.success("Successfully read document", icon='✅')
    divider()

    prompt = st.text_area(
        "Enter your prompt:",
        value=user_prompt_1, height=600,
        help="Below is the current prompt being used"
    )

    analyze = st.button("Analyze document using prompt")

    if analyze:
        if document is not None:
            source_ref, response = get_prompted_result(document, prompt)
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
                    print(prompt, file=f)
        else:
            st.warning("Please upload a document or use the preloaded document", icon="⚠️")
