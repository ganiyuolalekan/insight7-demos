import streamlit as st

from app_utils import app_meta, divider
from documentation import introduction
from st_funcs import theme_api, frequency_api


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
        test_theme_api = st.checkbox(
            label="Test the theme-based API Layer",
            help="Test prompt API",
            value=False
        )
        divider()
        test_frequency_api = st.checkbox(
            label="Test the frequency-based API Layer",
            help="Test prompt API",
            value=False
        )

    if test_theme_api:
        theme_api()
    elif test_frequency_api:
        frequency_api()
    else:
        st.write(introduction)

else:
    # Production
    with open('insight_highlights/README.md', 'r') as f:
        demo_report = f.read()
    st.markdown(demo_report)

    # Local
    # with open('README.md', 'r') as f:
    #     demo_report = f.read()
    # st.markdown(demo_report)
