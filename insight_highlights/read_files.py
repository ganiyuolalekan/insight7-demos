import io
import PyPDF2
import docx2txt
import pandas as pd

THRESH = 80


def is_significant_text(text):
    """Check if the text contains a significant amount of alphanumeric characters"""

    return sum(1 for char in text if char.isalnum()) >= THRESH


def identify_text_columns(df, threshold):
    """
    Given a pandas dataframe and a threshold percentage for identifying significant text columns,
    returns a list of object columns with at least the specified percentage of text values,
    excluding any columns that only contain email addresses or do not contain a majority of sentences
    with at least 3 words in a majority of its rows.
    """
    text_cols = []
    for col in df.select_dtypes(include='object'):
        # Check if column contains only email addresses
        if df[col].str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$').all():
            continue
        # Calculate percentage of text values in column
        text_pct = df[col].str.count('[a-zA-Z]').sum() / df[col].str.len().sum()
        if text_pct < threshold:
            continue
        # Check if column contains majority of sentences with at least 3 words in a majority of its rows
        sentence_count = df[col].str.count('[.!?]')
        word_count = df[col].str.count('\w+')
        sentence_lengths = word_count / sentence_count.replace(0, 1)
        sentence_pct = (sentence_lengths >= 1).sum() / sentence_count.count()
        if sentence_pct < threshold:
            continue
        text_cols.append(col)
    return text_cols


def present_csv_file(df):
    data = df[identify_text_columns(df, .7)]
    if len(data.columns):
        data = data.values.tolist()
        result = []
        for d in data:
            if type(d) == list:
                result.append(' - '.join(list(map(str, d))))
            else:
                result.append(d)

        return '\n\n'.join(result)
    else:
        raise ValueError("CSV file does not contain any column with usable text data")


def read_uploaded_file(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        return present_csv_file(pd.read_csv(file))
    elif file_extension == 'docx':
        return docx2txt.process(file)
    elif file_extension == 'pdf':
        return ''.join([
            page.extract_text()
            for page in PyPDF2.PdfReader(io.BytesIO(file.read())).pages
        ])
    elif file_extension == 'txt':
        return file.read().decode('utf-8')
    else:
        return "File type not supported. Please upload a file with extension csv, docx, pdf, or txt."
