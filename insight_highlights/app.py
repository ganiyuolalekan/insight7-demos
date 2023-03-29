import streamlit as st

from st_funcs import doc_correction, prompt_eng
from documentation import introduction
from utils import app_meta, divider


app_meta()

# Initializing Side-Bar
with st.sidebar:
    st.write("Insight7 API tester")
    start_project = st.checkbox(
        label="Start Application",
        help="Starts The Demo Application"
    )
    divider()

if start_project:
    with st.sidebar:
        test_document_correction = st.checkbox(
            label="Test the document fine-tuning Layer",
            help="Test fine-tuning API",
            value=False
        )
        divider()
        test_prompt = st.checkbox(
            label="Test the prompt Layer",
            help="Test prompt API",
            value=False
        )

    if test_document_correction:
        doc_correction()
    elif test_prompt:
        prompt_eng()
    else:
        st.write(introduction)

else:
    with open('insight_highlights/README.md', 'r') as f:
        demo_report = f.read()
    st.markdown(demo_report)
