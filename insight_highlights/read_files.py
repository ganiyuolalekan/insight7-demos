import io
import PyPDF2
import docx2txt
import pandas as pd


def read_uploaded_file(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        return pd.read_csv(file).to_string(index=False)
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
