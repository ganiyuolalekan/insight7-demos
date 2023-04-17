# Application Utility Functions

import streamlit as st


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
