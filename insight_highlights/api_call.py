import requests
# from transformers import pipeline
# from happytransformer import HappyTextToText, TTSettings


def file_extractor(file_url):
    """Calls the file extractor API to get extracted content"""

    return eval(requests.request(
        "POST",
        "https://file-extractor-azi5xvdx6a-ue.a.run.app/read",
        data={'file_url': file_url}
    ).text)['body']


# def clean_text(doc):
#     """Hugging Face API for grammar correction"""
#
#     return HappyTextToText(
#         "T5", "vennify/t5-base-grammar-correction"
#     ).generate_text(
#         doc, args=TTSettings(num_beams=5, min_length=1)
#     ).text
#
#
# def correct_grammar(doc):
#     """"Hugging Face API for grammar correction"""
#
#     return pipeline(
#         'text2text-generation',
#         'pszemraj/flan-t5-large-grammar-synthesis',
#     )(doc)
