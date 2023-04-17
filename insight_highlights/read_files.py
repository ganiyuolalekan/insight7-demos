import io
import csv
import PyPDF2
import docx2txt

THRESH = 80


def is_significant_text(text):
    """Check if the text contains a significant amount of alphanumeric characters"""

    return sum(1 for char in text if char.isalnum()) >= THRESH


def read_uploaded_file(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        rows = [
            ' '.join(list(map(str, row)))
            for i, row in enumerate(
                csv.reader(io.TextIOWrapper(file))
            )
            if is_significant_text(' '.join(list(map(str, row))))
        ]
        return '\n\n'.join(rows[1:])
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
