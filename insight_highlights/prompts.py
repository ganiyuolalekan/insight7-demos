from langchain.prompts import PromptTemplate


template = """Create an answer to the given questions using the provided document excerpts (in no particular order) as references. Include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. Always ensure all included sources are from your documents. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty.
---------
QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.\
While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.\
The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. (SOURCES: 1-32)
---------
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

query = """Please analyze the given document and generate a report that includes pain points, desires, and behaviors, grouped by category. For each pain point, desire, and behavior, please also include the source/reference from the document.

In the report, please use the following format:

Pain Points
<pain point insights> (<source/reference>)
...
Desires
<desire insights> (<source/reference>)
...
Behaviours
<behaviour insights> (<source/reference>)
...

Please ensure that each insight is concise and relevant to the category it belongs to. Also, please frame the insights to have a tone that communicates to a manager or third party reader, ensuring it is relevant to readers."""

STUFF_PROMPT = PromptTemplate(
    template=template,
    input_variables=["summaries", "question"]
)

user_prompt_1 = """Please analyze the insights report provided and categorize the pain points, desires, and behaviors into topics/themes. For each topic, please include a descriptive name that reflects the main theme of the category, and group the associated pain points, desires, and behaviors under that topic.
Please use the following format to organize the results into a JSON format:
{
    "<topic name 1>": {
        "Pain Points": [
            ["<point>", "<point source>"],
            ...
        ],
        "Desires": [
            ["<point>", "<point source>"],
            ...
        ],
        "Behaviours": [
            ["<point>", "<point source>"],
            ...
        ]
    },
    ...,
}
Make sure that each pain point, desire, and behavior is associated with the relevant category and that the topics are clearly defined and descriptive of the points they contain. Please also ensure that each point is properly formatted with its source."""

user_prompt_2 = lambda source_ref: f"""
After that replace each <point source> with it's corresponding text in {source_ref}

For each pair of ["<point>", "<point source>"] extract the referencing text in <point source> that depicts the idea in <point>, and replace 
<point source> with the extracted referencing text. Ensure context is given to the extracted referencing text so it has volume. 
Also ensure the grammar is corrected and it is easy to read and digest."""

i_user_prompt = lambda source_ref: user_prompt_1 + user_prompt_2(source_ref)

clean_document_prompt = """Given the following text, perform grammar correction on the text"""

points = lambda tags: (', '.join(tags[:-1]) + ' and ' + tags[-1]).lower()
def json_points(tags):
    last_tag = tags[-1]
    result = """{
    "<topic name 1>": {
        """
    for tag in tags:
        if tag == last_tag:
            result += f""""{' '.join([_tag.capitalize() for _tag in tag.split(' ')])}": [
                ["<point>", "<point source>"],
                ...
            ],
        """
        else:
            result += f""""{' '.join([_tag.capitalize() for _tag in tag.split(' ')])}": [
                ["<point>", "<point source>"],
                ...
            ],
            """

    result += """},
    ...,
}"""

    return result


def query_points(tags):
    result = """"""

    for tag in tags:
        result += f"""{' '.join([_tag.capitalize() for _tag in tag.split(' ')])}
<pain point insights> (<source/reference>)
...
"""

    return result


user_prompt_for_tags = lambda tags: f"""Please analyze the insights report provided and categorize the {points(tags)} into topics/themes. For each topic, please include a descriptive name that reflects the main theme of the category, and group the associated pain points, desires, and behaviors under that topic.
Please use the following format to organize the results into a JSON format:
{json_points(tags)}
Make sure that each pain point, desire, and behavior is associated with the relevant category and that the topics are clearly defined and descriptive of the points they contain. Please also ensure that each point is properly formatted with its source."""

query_for_tags = lambda tags: f"""Please analyze the given document and generate a report that includes {points(tags)}, grouped by category. For each {points(tags)}, please also include the source/reference from the document.
In the report, please use the following format:
{query_points(tags)}
Please ensure that each insight is concise and relevant to the category it belongs to. Also, please frame the insights to have a tone that communicates to a manager or third party reader, ensuring it is relevant to readers."""
