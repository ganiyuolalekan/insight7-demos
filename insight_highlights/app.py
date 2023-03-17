import pickle
import streamlit as st

from io import StringIO

from utils import app_meta, divider, generate_insight_api, generate_insights_highlights


app_meta()

# Initializing Side-Bar
with st.sidebar:
    st.write("Frequency Based Themes")
    start_project = st.checkbox(
        label="Start Application",
        help="Starts The Demo Application"
    )
    divider()

if start_project:
    with st.sidebar:
        st.write("Start by imputing three (3) documents and the algorithm will handle the rest")
        divider()

        use_test_output = st.checkbox(
            label="Use an example output for quicker experiment",
            help="Quickly evaluate functions",
            value=True
        )
        st.write("> Helps to quickly demo the app function")
        divider()

        theme_similarity_thresh = st.slider(
            "Theme Similarity Thresh", min_value=.0, max_value=1.00, value=.20, step=.001,
            help="The greater the theme similarity, the more strict the insights picked"
        )
        st.markdown("> This thresh is intentionally set low to capture multiple similar insights")
        st.markdown("> If you believe the document could contain related insights, feel free to increase the threshold")
        divider()

        st.markdown("> Also note: The program was hardcoded to take only three file inputs for simplicity")
        divider()

        document_1 = st.file_uploader("Choose your first document file")
        document_2 = st.file_uploader("Choose your second document file")
        document_3 = st.file_uploader("Choose your third document file")
        divider()

        analyse = st.button('Analyse Documents')
        divider()

    if use_test_output:
        document_data = {
            i: open(file_path, 'r').read()
            for i, file_path in enumerate((
                "insight_highlights/data/doc1.txt",
                "insight_highlights/data/doc3.txt",
                "insight_highlights/data/data.txt"
            ), start=1)
        }
    else:
        document_data = {
            i: StringIO(document.getvalue().decode("utf-8")).read()
            for i, document in enumerate((document_1, document_2, document_3), start=1)
            if document is not None
        }

    if len(document_data) < 3:
        st.write("### Please upload the document files to begin")
        st.write(
            "Please ensure your files are .txt files. this program isn't sophisticated enough for other file formats")
    else:
        st.write("# Preview your document here")

        tab1, tab2, tab3 = st.tabs(["First document", "Second document", "Third document"])

        with tab1:
            st.header("Here's the data of the first document")
            st.write(document_data[1])

        with tab2:
            st.header("Here's the data of the second document")
            st.write(document_data[2])

        with tab3:
            st.header("Here's the data of the third document")
            st.write(document_data[3])

        divider()

        if analyse:
            if use_test_output:
                document_api_result = pickle.load(open('data/document_api_result.p', 'rb'))
            else:
                document_api_result = {
                    d: eval(generate_insight_api(document_data[d]))
                    for d in document_data
                }

            st.write("# Result from the current API on the documents")

            _tab1, _tab2, _tab3 = st.tabs(["First document", "Second document", "Third document"])

            with _tab1:
                st.header("Here's API call for the first document")
                st.write(document_api_result[1])

            with _tab2:
                st.header("Here's API call for the second document")
                st.write(document_api_result[2])

            with _tab3:
                st.header("Here's API call for the third document")
                st.write(document_api_result[3])

            divider()
            st.write("""The current frequency based theme makes use of this format which is likely to change
            {
                'data': {
                    '<tag>': {
                        '<insight>': ['[<highlights>]', '<doc>'],
                        '<highlight-count>': '<count>'
                    }
                    ...
                }
            }
            """)
            st.write('Click the "Generate the frequency based themes" button to generate the frequency based result')

            divider()

            if use_test_output:
                data = pickle.load(open('data/data.p', 'rb'))
            else:
                data = {
                    'data': {
                        "doc_1": document_api_result[1]['data'],
                        "doc_2": document_api_result[2]['data'],
                        "doc_3": document_api_result[3]['data'],
                    }
                }

            result = generate_insights_highlights(
                data,
                ['Pain Points', 'Behaviour', 'Desires'],
                theme_similarity_thresh
            )

            st.write("# Mapping insights to a theme based on similarity")

            st.write(result['data']['Pain Points'])

            __tab1, __tab2, __tab3, __tab4 = st.tabs([
                "JSON output", "Pain Points",
                "Desires", "Behaviour"
            ])

            with __tab1:
                st.header("Here's an example JSON output")
                st.write(result)

            with __tab2:
                st.header("Here are the insights on the 'Pain Points'")
                points = result['data']['Pain Points']
                for insight in list(points)[:-1]:
                    st.write(f"#### '{insight}' has the following highlights")
                    for i, highlight in enumerate(points[insight]):
                        st.write(i, highlight)

            with __tab3:
                st.header("Here are the insights on the 'Desires'")
                points = result['data']['Desires']
                for insight in list(points)[:-1]:
                    st.write(f"#### '{insight}' has the following highlights")
                    for i, highlight in enumerate(points[insight]):
                        st.write(i, highlight)

            with __tab4:
                st.header("Here are the insights on the 'Behaviour'")
                points = result['data']['Behaviour']
                for insight in list(points)[:-1]:
                    st.write(f"#### '{insight}' has the following highlights")
                    for i, highlight in enumerate(points[insight]):
                        st.write(i, highlight)

else:
    with open('insight_highlights/README.md', 'r') as f:
        demo_report = f.read()
    st.markdown(demo_report)
