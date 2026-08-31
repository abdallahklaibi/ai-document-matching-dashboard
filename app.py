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

def normalize_phone(value):
    if value is None or pd.isna(value):
        return ""
    s = re.sub(r"\D", "", str(value))
    if not s:
        return ""
    # Remove common international dialing prefix 00.
    if s.startswith("00"):
        s = s[2:]
    return s

def phone_variants(phone):
    p = normalize_phone(phone)
    if not p:
        return set()
    variants = {p}
    # Useful matching forms when source documents differ in country-code formatting.
    if len(p) >= 7:
        variants.add(p[-7:])
    if len(p) >= 8:
        variants.add(p[-8:])
    if len(p) >= 9:
        variants.add(p[-9:])
    return variants

def detect_phone_column(df):
    preferred = [
        "phone", "mobile", "telephone", "tel", "contact", "contact number",
        "phone number", "mobile number", "number"
    ]
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for p in preferred:
        if p in normalized:
            return normalized[p]

    # Heuristic: column with the highest share of phone-like values.
    best_col, best_score = None, 0
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(200)
        if len(sample) == 0:
            continue
        scores = []
        for v in sample:
            digits = re.sub(r"\D", "", v)
            scores.append(7 <= len(digits) <= 16)
        score = sum(scores) / max(len(scores), 1)
        if score > best_score:
            best_col, best_score = col, score
    return best_col if best_score >= 0.35 else None

def extract_text_pdf(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        parts.append(f"\n--- PAGE {i} ---\n{text}")
    return "\n".join(parts), len(reader.pages)

def extract_phone_candidates(text):
    # Broad candidate pattern; normalization + length checks happen afterwards.
    raw = re.findall(r"(?:\+?\d[\d\s()./-]{5,}\d)", text or "")
    seen, result = set(), []
    for item in raw:
        p = normalize_phone(item)
        if 7 <= len(p) <= 16 and p not in seen:
            seen.add(p)
            result.append(p)
    return result

def parse_json_from_text(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # recover JSON array/object embedded in prose/fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if m:
        return json.loads(m.group(1))
    raise ValueError("AI response did not contain valid JSON.")

def claude_extract_phones(pdf_bytes):
    api_key = get_secret("ANTHROPIC_API_KEY")
    model = get_secret("CLAUDE_MODEL", "claude-sonnet-5")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
    if Anthropic is None:
        raise RuntimeError("The anthropic Python package is not installed.")

    client = Anthropic(api_key=api_key)
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    prompt = """
You are reading a monthly business document. Extract every phone/mobile/telephone
number that is visibly present in the PDF, including handwritten numbers where
you can read them.

Return ONLY a JSON array. Each element must have:
- phone_raw: exactly what you can read
- phone_normalized: digits only
- page: page number if known, otherwise null
- confidence: one of "High", "Medium", "Low"

Do not invent missing digits. If uncertain, keep the visible number and mark Low.
Deduplicate only exact duplicates on the same page.
"""
    msg = client.messages.create(
        model=model,
        max_tokens=12000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = "".join(getattr(block, "text", "") for block in msg.content if getattr(block, "type", "") == "text")
    data = parse_json_from_text(text)
    if isinstance(data, dict):
        data = data.get("phones", data.get("results", []))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list from Claude.")
    return data

def build_reference_index(df, phone_col):
    rows = []
    for idx, value in df[phone_col].items():
        normalized = normalize_phone(value)
        if normalized:
            rows.append((idx, normalized, phone_variants(normalized)))
    return rows

def find_best_match(source_phone, reference_index):
    src = normalize_phone(source_phone)
    if not src:
        return None, "Not Matched", 0

    src_vars = phone_variants(src)
    exact_matches = [(idx, ref) for idx, ref, _ in reference_index if ref == src]
    if exact_matches:
        return exact_matches[0][0], "High", 100

    # Country-code / formatting tolerant match.
    candidates = []
    for idx, ref, ref_vars in reference_index:
        common = src_vars.intersection(ref_vars)
        common_lengths = [len(x) for x in common if len(x) >= 7]
        if common_lengths:
            strongest = max(common_lengths)
            score = 95 if strongest >= 9 else (88 if strongest == 8 else 78)
            candidates.append((score, idx, ref))

    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        score, idx, _ = candidates[0]
        confidence = "High" if score >= 90 else ("Medium" if score >= 80 else "Low")
        return idx, confidence, score

    return None, "Not Matched", 0

def create_output_excel(original_df, results_df, matched_reference_rows):
    output = io.BytesIO()
    export_df = original_df.copy()

    export_df["AI_Match_Status"] = "Not Matched"
    export_df["AI_Confidence"] = ""
    export_df["AI_Source_Phone"] = ""
    export_df["AI_Source_Page"] = ""

    for ref_idx, result_rows in matched_reference_rows.items():
        best = sorted(result_rows, key=lambda r: r["Match Score"], reverse=True)[0]
        export_df.loc[ref_idx, "AI_Match_Status"] = "Matched"
        export_df.loc[ref_idx, "AI_Confidence"] = best["Confidence"]
        export_df.loc[ref_idx, "AI_Source_Phone"] = best["PDF Phone"]
        export_df.loc[ref_idx, "AI_Source_Page"] = best.get("Page", "")

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Matched Data")
        results_df.to_excel(writer, index=False, sheet_name="AI Audit")

    output.seek(0)
    wb = load_workbook(output)
    ws = wb["Matched Data"]

    yellow = PatternFill("solid", fgColor="FFF2CC")
    orange = PatternFill("solid", fgColor="FCE4D6")
    green = PatternFill("solid", fgColor="E2F0D9")
    gray = PatternFill("solid", fgColor="E7E6E6")

    headers = {cell.value: cell.column for cell in ws[1]}
    conf_col = headers.get("AI_Confidence")
    status_col = headers.get("AI_Match_Status")

    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="center")

    for row in range(2, ws.max_row + 1):
        status = ws.cell(row, status_col).value if status_col else ""
        conf = ws.cell(row, conf_col).value if conf_col else ""
        fill = None
        if status == "Matched" and conf == "High":
            fill = green
        elif status == "Matched" and conf == "Medium":
            fill = yellow
        elif status == "Matched" and conf == "Low":
            fill = orange
        else:
            fill = gray
        if fill:
            for col in range(1, ws.max_column + 1):
                ws.cell(row, col).fill = fill

    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for col_cells in sheet.columns:
            max_len = min(max(len(str(c.value or "")) for c in col_cells) + 2, 40)
            sheet.column_dimensions[col_cells[0].column_letter].width = max_len

    final = io.BytesIO()
    wb.save(final)
    final.seek(0)
    return final.getvalue()

# ---------- UI ----------
st.markdown("""
<div class="hero">
  <h1>AI Document Matching & Reconciliation</h1>
  <p>Monthly PDF-to-Excel matching with confidence scoring, review controls and an auditable output.</p>
</div>
""", unsafe_allow_html=True)

steps = st.columns(4)
for c, title, desc in zip(
    steps,
    ["1 · Upload", "2 · Analyse", "3 · Review", "4 · Export"],
    ["PDF + Excel", "AI-assisted extraction", "Confidence & exceptions", "Highlighted Excel"],
):
    c.markdown(f'<div class="step"><b>{title}</b><br><span class="small-muted">{desc}</span></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns(2)
with left:
    pdf_file = st.file_uploader("Monthly PDF", type=["pdf"], help="Upload the monthly source PDF.")
with right:
    excel_file = st.file_uploader("Monthly Excel", type=["xlsx", "xls"], help="Upload the reference Excel file.")

df = None
phone_col = None
if excel_file is not None:
    try:
        df = pd.read_excel(excel_file)
        phone_col = detect_phone_column(df)
        st.success(f"Excel loaded: {len(df):,} rows")
        if phone_col:
            st.caption(f"Detected phone column: **{phone_col}**")
        else:
            st.warning("I could not confidently detect a phone-number column.")
    except Exception as e:
        st.error(f"Could not read Excel: {e}")

if df is not None and len(df.columns):
    chosen = st.selectbox(
        "Excel phone-number column",
        options=list(df.columns),
        index=list(df.columns).index(phone_col) if phone_col in df.columns else 0,
    )
    phone_col = chosen

api_ready = bool(get_secret("ANTHROPIC_API_KEY"))
mode_options = ["Standard PDF text extraction"]
if api_ready:
    mode_options.insert(0, "Claude AI document reading (recommended for scans/handwriting)")
analysis_mode = st.radio("Processing method", mode_options, horizontal=True)
if not api_ready:
    st.info("Claude AI is not configured yet. The dashboard will still work for PDFs containing selectable text.")

run = st.button(
    "Run Monthly Analysis",
    type="primary",
    use_container_width=True,
    disabled=(pdf_file is None or df is None or phone_col is None),
)

if run:
    pdf_bytes = pdf_file.getvalue()
    reference_index = build_reference_index(df, phone_col)

    progress = st.progress(0, text="Preparing files…")
    try:
        page_count = None
        extracted = []

        if analysis_mode.startswith("Claude"):
            progress.progress(15, text="Claude is reading the PDF…")
            extracted = claude_extract_phones(pdf_bytes)
            # normalize expected fields
            source_records = []
            for item in extracted:
                raw = item.get("phone_raw", item.get("phone", ""))
                norm = normalize_phone(item.get("phone_normalized", raw))
                if 7 <= len(norm) <= 16:
                    source_records.append({
                        "PDF Phone": norm,
                        "Page": item.get("page"),
                        "Read Confidence": item.get("confidence", "Medium"),
                    })
        else:
            progress.progress(15, text="Reading PDF text…")
            text, page_count = extract_text_pdf(pdf_bytes)
            phones = extract_phone_candidates(text)
            source_records = [{"PDF Phone": p, "Page": None, "Read Confidence": "High"} for p in phones]

        progress.progress(55, text="Matching against Excel…")

        result_rows = []
        matched_reference_rows = {}
        for rec in source_records:
            ref_idx, confidence, score = find_best_match(rec["PDF Phone"], reference_index)
            matched = ref_idx is not None
            row = {
                "PDF Phone": rec["PDF Phone"],
                "Page": rec.get("Page"),
                "Read Confidence": rec.get("Read Confidence", ""),
                "Status": "Matched" if matched else "Not Matched",
                "Confidence": confidence,
                "Match Score": score,
                "Excel Row": int(ref_idx) + 2 if matched else None,
                "Excel Phone": normalize_phone(df.loc[ref_idx, phone_col]) if matched else "",
            }
            result_rows.append(row)
            if matched:
                matched_reference_rows.setdefault(ref_idx, []).append(row)

        results = pd.DataFrame(result_rows)
        progress.progress(80, text="Preparing audit report…")
        output_bytes = create_output_excel(df, results, matched_reference_rows)
        progress.progress(100, text="Analysis complete")

        st.session_state["results"] = results
        st.session_state["output_bytes"] = output_bytes
        st.session_state["source_count"] = len(source_records)
        st.session_state["excel_count"] = len(df)
        st.session_state["pages"] = page_count

    except Exception as e:
        progress.empty()
        st.error(f"Analysis failed: {e}")

if "results" in st.session_state:
    results = st.session_state["results"]
    total = len(results)
    matched = int((results["Status"] == "Matched").sum()) if total else 0
    review = int(results["Confidence"].isin(["Medium", "Low"]).sum()) if total else 0
    rate = (matched / total * 100) if total else 0

    st.divider()
    st.subheader("Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numbers Detected", f"{total:,}")
    c2.metric("Matched", f"{matched:,}")
    c3.metric("Manual Review", f"{review:,}")
    c4.metric("Match Rate", f"{rate:.1f}%")

    if total == 0:
        st.warning("No phone numbers were detected in the PDF. If this is a scanned or handwritten PDF, configure Claude AI mode.")
    else:
        st.subheader("Match Results")
        f1, f2 = st.columns([1, 2])
        with f1:
            confidence_filter = st.multiselect(
                "Confidence",
                options=["High", "Medium", "Low", "Not Matched"],
                default=["High", "Medium", "Low", "Not Matched"],
            )
        with f2:
            search = st.text_input("Search phone / Excel row")

        filtered = results[results["Confidence"].isin(confidence_filter)].copy()
        if search:
            mask = filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            filtered = filtered[mask]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Match Score": st.column_config.ProgressColumn("Match Score", min_value=0, max_value=100, format="%d%%"),
            },
        )

        stamp = datetime.now().strftime("%Y-%m-%d")
        st.download_button(
            "Download Highlighted Excel Report",
            data=st.session_state["output_bytes"],
            file_name=f"AI_Matched_Report_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

    with st.expander("How confidence is interpreted"):
        st.markdown("""
- **High** — exact or very strong phone-number match.
- **Medium** — strong match after formatting/country-code normalization.
- **Low** — partial match worth checking manually.
- **Not Matched** — no reliable Excel match found.

The downloadable workbook includes both the highlighted reference data and a separate **AI Audit** sheet.
        """)

st.divider()
st.caption("AI-assisted matching supports review; final business decisions remain with the responsible reviewer.")
