import requests


def file_extractor(file_url):
    """Calls the file extractor API to get extracted content"""

    return eval(requests.request(
        "POST",
        "https://file-extractor-azi5xvdx6a-ue.a.run.app/read",
        data={'file_url': file_url}
    ).text)['body']
