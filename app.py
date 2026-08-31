import io
import os
import re
import json
import base64
from datetime import datetime

import pandas as pd
import streamlit as st
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from pypdf import PdfReader

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

st.set_page_config(
    page_title="AI Document Matching Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {
    padding: 24px 28px;
    border: 1px solid rgba(120,120,120,.20);
    border-radius: 18px;
    margin-bottom: 22px;
}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.5rem 0 0 0; opacity:.75;}
.step {
    border: 1px solid rgba(120,120,120,.22);
    border-radius: 14px;
    padding: 14px 16px;
    min-height: 85px;
}
.small-muted {opacity:.67; font-size:.9rem;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(120,120,120,.20);
    padding: 16px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


def clean_text(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    s = str(value).upper()
    s = re.sub(r"[^A-Z0-9\u0600-\u06FF ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_phone(value):
    if value is None:
        return ""
    s = re.sub(r"\D", "", str(value))
    if s.startswith("00971"):
        s = "0" + s[5:]
    elif s.startswith("971"):
        s = "0" + s[3:]
    return s

def phone_variants(value):
    p = normalize_phone(value)
    if not p:
        return set()
    out = {p}
    for n in (7, 8, 9, 10):
        if len(p) >= n:
            out.add(p[-n:])
    return out

def normalize_account(value):
    if value is None:
        return ""
    s = re.sub(r"\D", "", str(value))
    return s.lstrip("0") or s

def similarity(a, b):
    from difflib import SequenceMatcher
    a, b = clean_text(a), clean_text(b)
    if not a or not b:
        return 0
    return int(round(SequenceMatcher(None, a, b).ratio() * 100))

def parse_json_from_text(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if m:
            return json.loads(m.group(1))
    raise ValueError("Claude response did not contain valid JSON.")

def claude_extract_customer_fields(pdf_bytes, max_pages=None):
    """Claude reads the form visually. Priority is phone -> name -> account."""
    api_key = get_secret("ANTHROPIC_API_KEY")
    model = get_secret("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured in Streamlit Secrets.")
    if Anthropic is None:
        raise RuntimeError("The anthropic Python package is not installed.")

    # For quick testing, create a small PDF containing only the first N pages.
    use_bytes = pdf_bytes
    page_count = None
    if max_pages:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        writer = PdfWriter()
        for page in reader.pages[:max_pages]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        use_bytes = buf.getvalue()

    client = Anthropic(api_key=api_key)
    pdf_b64 = base64.b64encode(use_bytes).decode("utf-8")

    prompt = """
Read each Customer Satisfaction and Service Evaluation form visually.

For EACH PDF page, read ONLY the customer-identification fields near the top:
1) Telephone Number
2) Name
3) Account/reference number if one is written near the customer details,
   including a number written in parentheses.

IMPORTANT:
- Handwriting is expected.
- Do not copy the printed form title, printed labels, evaluation questions,
  dates, signatures, or unrelated numbers.
- Never invent missing characters or digits.
- If a field cannot be read reliably, return an empty string for that field.
- Keep page numbers aligned with the PDF.

Return ONLY a JSON array with exactly one object per page:
[
  {
    "page": 1,
    "phone": "",
    "name": "",
    "account": "",
    "phone_confidence": "High|Medium|Low|Unreadable",
    "name_confidence": "High|Medium|Low|Unreadable",
    "account_confidence": "High|Medium|Low|Unreadable"
  }
]
"""
    msg = client.messages.create(
        model=model,
        max_tokens=12000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                }},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = "".join(
        getattr(block, "text", "")
        for block in msg.content
        if getattr(block, "type", "") == "text"
    )
    data = parse_json_from_text(text)
    if isinstance(data, dict):
        data = data.get("results", data.get("pages", []))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list from Claude.")
    return data, page_count

def extract_phones_from_cell(value):
    if value is None or pd.isna(value):
        return []
    text = str(value)
    chunks = re.findall(r"(?:\+?971|00971|0)?\s*\d[\d\s().-]{6,}\d", text)
    phones = []
    for x in chunks:
        p = normalize_phone(x)
        if 7 <= len(p) <= 12:
            phones.append(p)
    # Fallback for a simple single number.
    if not phones:
        p = normalize_phone(text)
        if 7 <= len(p) <= 12:
            phones.append(p)
    return list(dict.fromkeys(phones))

def choose_default(columns, preferred):
    lower = {str(c).strip().lower(): c for c in columns}
    for p in preferred:
        if p.lower() in lower:
            return lower[p.lower()]
    return columns[0]

def match_record(rec, df, phone_col, name_col, account_col):
    """
    User-requested fallback:
    1. clear phone that matches
    2. otherwise clear name
    3. otherwise account number
    Evidence from other fields can raise confidence.
    """
    pdf_phone = normalize_phone(rec.get("phone", ""))
    pdf_name = clean_text(rec.get("name", ""))
    pdf_account = normalize_account(rec.get("account", ""))

    phone_conf = str(rec.get("phone_confidence", "")).lower()
    name_conf = str(rec.get("name_confidence", "")).lower()
    acct_conf = str(rec.get("account_confidence", "")).lower()

    # ---- 1) PHONE FIRST ----
    if pdf_phone and phone_conf in ("high", "medium"):
        src_vars = phone_variants(pdf_phone)
        candidates = []
        for idx, val in df[phone_col].items():
            for ep in extract_phones_from_cell(val):
                common = src_vars & phone_variants(ep)
                if any(len(x) >= 8 for x in common):
                    candidates.append(idx)
                    break
        if len(candidates) == 1:
            return candidates[0], "Phone", "High", 100
        if len(candidates) > 1:
            # Resolve duplicate phone using name/account if possible.
            best = None
            for idx in candidates:
                ns = similarity(pdf_name, df.at[idx, name_col]) if pdf_name else 0
                am = bool(pdf_account and normalize_account(df.at[idx, account_col]) == pdf_account)
                score = 100 + (15 if am else 0) + int(ns * .15)
                if best is None or score > best[0]:
                    best = (score, idx)
            return best[1], "Phone + confirmation", "High", min(best[0], 100)

    # ---- 2) NAME FALLBACK ----
    if pdf_name and name_conf in ("high", "medium"):
        scored = []
        for idx, val in df[name_col].items():
            sc = similarity(pdf_name, val)
            if sc >= 62:
                scored.append((sc, idx))
        scored.sort(reverse=True)
        if scored:
            best_score, best_idx = scored[0]
            # Avoid accepting an ambiguous name.
            gap = best_score - scored[1][0] if len(scored) > 1 else best_score
            if best_score >= 82 and gap >= 4:
                return best_idx, "Name", "High" if best_score >= 90 else "Medium", best_score
            if best_score >= 72 and gap >= 7:
                return best_idx, "Name", "Medium", best_score

    # ---- 3) ACCOUNT FALLBACK ----
    if pdf_account and acct_conf in ("high", "medium"):
        matches = []
        for idx, val in df[account_col].items():
            if normalize_account(val) == pdf_account:
                matches.append(idx)
        if len(matches) == 1:
            return matches[0], "Account Number", "High", 100
        if len(matches) > 1:
            return matches[0], "Account Number", "Medium", 90

    return None, "No reliable match", "Not Matched", 0

def create_output_excel(original_df, results_df, matched_reference_rows):
    output = io.BytesIO()
    export_df = original_df.copy()

    for col, default in [
        ("AI_Match_Status", "Not Matched"),
        ("AI_Confidence", ""),
        ("AI_Match_Method", ""),
        ("AI_Source_Phone", ""),
        ("AI_Source_Name", ""),
        ("AI_Source_Account", ""),
        ("AI_Source_Page", ""),
    ]:
        export_df[col] = pd.Series([default] * len(export_df), index=export_df.index, dtype="object")

    for ref_idx, result_rows in matched_reference_rows.items():
        best = sorted(result_rows, key=lambda r: r["Match Score"], reverse=True)[0]
        export_df.loc[ref_idx, "AI_Match_Status"] = "Matched"
        export_df.loc[ref_idx, "AI_Confidence"] = best["Confidence"]
        export_df.loc[ref_idx, "AI_Match_Method"] = best["Match Method"]
        export_df.loc[ref_idx, "AI_Source_Phone"] = best["PDF Phone"]
        export_df.loc[ref_idx, "AI_Source_Name"] = best["PDF Name"]
        export_df.loc[ref_idx, "AI_Source_Account"] = best["PDF Account"]
        export_df.loc[ref_idx, "AI_Source_Page"] = str(best["Page"])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Matched Data")
        results_df.to_excel(writer, index=False, sheet_name="AI Audit")

    output.seek(0)
    wb = load_workbook(output)
    green = PatternFill("solid", fgColor="E2F0D9")
    yellow = PatternFill("solid", fgColor="FFF2CC")
    gray = PatternFill("solid", fgColor="E7E6E6")

    ws = wb["Matched Data"]
    headers = {cell.value: cell.column for cell in ws[1]}
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in range(2, ws.max_row + 1):
        status = ws.cell(r, headers["AI_Match_Status"]).value
        conf = ws.cell(r, headers["AI_Confidence"]).value
        fill = green if status == "Matched" and conf == "High" else yellow if status == "Matched" else gray
        for c in range(1, ws.max_column + 1):
            ws.cell(r, c).fill = fill

    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cells in sheet.columns:
            width = min(max(len(str(c.value or "")) for c in cells) + 2, 40)
            sheet.column_dimensions[cells[0].column_letter].width = width

    final = io.BytesIO()
    wb.save(final)
    return final.getvalue()

# ---------- UI ----------
st.markdown("""
<div class="hero">
  <h1>AI Document Matching & Reconciliation</h1>
  <p>Claude reads Phone → Name → Account Number, then reconciles each PDF page against Excel.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    pdf_file = st.file_uploader("Monthly PDF", type=["pdf"])
with right:
    excel_file = st.file_uploader("Monthly Excel", type=["xlsx", "xls"])

df = None
if excel_file is not None:
    try:
        df = pd.read_excel(excel_file)
        st.success(f"Excel loaded: {len(df):,} rows")
    except Exception as e:
        st.error(f"Could not read Excel: {e}")

if df is not None and len(df.columns):
    cols = list(df.columns)
    default_phone = choose_default(cols, ["woCompletedWorkOrderSitePhone", "phone", "mobile"])
    default_name = choose_default(cols, ["businessname", "COMBOname", "woCompletedSiteContact"])
    default_account = choose_default(cols, ["accountnum"])

    c1, c2, c3 = st.columns(3)
    with c1:
        phone_col = st.selectbox("Excel phone column", cols, index=cols.index(default_phone))
    with c2:
        name_col = st.selectbox("Excel name column", cols, index=cols.index(default_name))
    with c3:
        account_col = st.selectbox("Excel account-number column", cols, index=cols.index(default_account))

    test_mode = st.radio(
        "Run mode",
        ["Quick Test — first 10 pages", "Full Monthly Run — all pages"],
        horizontal=True,
    )

    if not get_secret("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY is missing from Streamlit Secrets.")

    run = st.button(
        "Run Monthly Analysis",
        type="primary",
        use_container_width=True,
        disabled=(pdf_file is None or not get_secret("ANTHROPIC_API_KEY")),
    )

    if run:
        pdf_bytes = pdf_file.getvalue()
        progress = st.progress(5, text="Preparing PDF for Claude…")
        try:
            max_pages = 10 if test_mode.startswith("Quick") else None
            progress.progress(15, text="Claude is reading customer fields…")
            extracted, original_page_count = claude_extract_customer_fields(pdf_bytes, max_pages=max_pages)

            progress.progress(60, text="Matching Phone → Name → Account Number…")
            result_rows = []
            matched_reference_rows = {}

            for rec in extracted:
                ref_idx, method, confidence, score = match_record(
                    rec, df, phone_col, name_col, account_col
                )
                matched = ref_idx is not None
                row = {
                    "Page": rec.get("page"),
                    "Match Method": method,
                    "PDF Phone": normalize_phone(rec.get("phone", "")),
                    "PDF Name": rec.get("name", ""),
                    "PDF Account": normalize_account(rec.get("account", "")),
                    "Phone Read Confidence": rec.get("phone_confidence", ""),
                    "Name Read Confidence": rec.get("name_confidence", ""),
                    "Account Read Confidence": rec.get("account_confidence", ""),
                    "Status": "Matched" if matched else "Not Matched",
                    "Confidence": confidence,
                    "Match Score": score,
                    "Excel Row": int(ref_idx) + 2 if matched else None,
                    "Excel Phone": str(df.at[ref_idx, phone_col]) if matched else "",
                    "Excel Name": str(df.at[ref_idx, name_col]) if matched else "",
                    "Excel Account": str(df.at[ref_idx, account_col]) if matched else "",
                }
                result_rows.append(row)
                if matched:
                    matched_reference_rows.setdefault(ref_idx, []).append(row)

            results = pd.DataFrame(result_rows)
            progress.progress(85, text="Preparing highlighted Excel report…")
            output_bytes = create_output_excel(df, results, matched_reference_rows)
            progress.progress(100, text="Analysis complete")

            st.session_state["results_v3"] = results
            st.session_state["output_v3"] = output_bytes
        except Exception as e:
            progress.empty()
            st.error(f"Analysis failed: {e}")

if "results_v3" in st.session_state:
    results = st.session_state["results_v3"]
    total = len(results)
    matched = int((results["Status"] == "Matched").sum()) if total else 0
    review = int((results["Confidence"] == "Medium").sum()) if total else 0
    rate = matched / total * 100 if total else 0

    st.divider()
    st.subheader("Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF Pages Reviewed", total)
    c2.metric("Matched Pages", matched)
    c3.metric("Manual Review", review)
    c4.metric("Match Rate", f"{rate:.1f}%")

    st.subheader("Page-by-Page Match Results")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match Score": st.column_config.ProgressColumn(
                "Match Score", min_value=0, max_value=100, format="%d%%"
            )
        },
    )

    stamp = datetime.now().strftime("%Y-%m-%d")
    st.download_button(
        "Download Highlighted Excel Report",
        data=st.session_state["output_v3"],
        file_name=f"AI_Matched_Report_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )

st.divider()
st.caption("Claude-assisted reconciliation. Review medium-confidence matches before final business use.")
