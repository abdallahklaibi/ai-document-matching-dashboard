# AI Document Matching & Reconciliation Dashboard

A Streamlit management dashboard for a recurring monthly workflow:

1. Upload a PDF.
2. Upload an Excel reference file.
3. Extract phone numbers from the PDF.
4. Match them against the Excel phone column.
5. Review confidence and exceptions.
6. Download a highlighted Excel workbook with an audit sheet.

## Local setup

Use Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser should open the dashboard automatically.

## AI / scanned PDF mode

For scanned PDFs and handwriting, configure an Anthropic API key.

Create:

`.streamlit/secrets.toml`

using `secrets.toml.example` as the template.

Do not commit your real secrets file to GitHub.

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload this project, but do not upload your real `.streamlit/secrets.toml`.
3. Go to Streamlit Community Cloud.
4. Create an app and select `app.py`.
5. Add `ANTHROPIC_API_KEY` and `CLAUDE_MODEL` in the Streamlit app's Secrets settings.
6. Deploy and use the generated permanent URL.

## Monthly operation

No code changes are needed each month. Open the dashboard, upload that month's PDF and Excel file, run the analysis, review exceptions, and download the output.

## Important production note

Before using confidential/customer documents in production, confirm your organization's approved hosting, data-residency, retention, and AI/API policies.
