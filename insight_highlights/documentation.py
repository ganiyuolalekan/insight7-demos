document_correction = """## Tests on the <b style="color: #7C5784; font-style: small-caps;">Data Cleaning Layer</b>

---

#### How it works
1. Starts by loading your file.
    - If a file url is entered, it uses the file-extractor API to load the content from the file URL.
2. You could also check ☑️ **"Use test document"**, to test with a preloaded document.
3. Once the document has been loaded, the program for cleaning the document is called on the imputed document.
4. Finally, the result from the cleaned document file is displayed along side the imputed one for comparison.

> Note: The file extraction API, only works for these extensions `.txt`, `.docx`, `.pdf`, `.csv` & `.xlsx`.
> Uploading files directly from the app currently only supports `.txt` files.
> I would be making progressive updates soon. The API currently used is the openai API."""

introduction = """## Insight7 API tester

---

### Demo app for testing and evaluating the Layers within the API

The following layers are the layers with the API, and they are as follows
- Transcript Grammar Fine-Tuning Layer. ✅
- Document Query Layer. ❌
- Prompt Engineering Layer. ✅
    - > **Currently tests just the theme-based API.**
- Frequency Based Generation Layer. ❌

> I would be making progressive updates on each layer based on specification.
"""

prompt_testing = """## Tests on the <b style="color: #7C5784; font-style: small-caps;">Prompt Layer</b>

---

#### How it works
1. Starts by loading your file.
    - If a file url is entered, it uses the file-extractor API to load the content from the file URL.
2. You could also check ☑️ **"Use test document"**, to test with a preloaded document.
3. After this you can enter your prompt to test the output.
3. Once the document has been loaded, the program for testing the prompt is launched when you click on the Analyze button.
4. After a while, you should get your output.

> Note: The file extraction API, only works for these extensions `.txt`, `.docx`, `.pdf`, `.csv` & `.xlsx`.
> Uploading files directly from the app currently only supports `.txt` files.
> The API currently used is the openai API."""
