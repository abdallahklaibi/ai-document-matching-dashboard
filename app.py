import io
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
except Exception:
    fitz = None
    pytesseract = None
    Image = None

st.set_page_config(
    page_title="Document Matching Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {padding: 24px 28px; border: 1px solid rgba(120,120,120,.20); border-radius: 18px; margin-bottom: 22px;}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.5rem 0 0 0; opacity:.75;}
.step {border: 1px solid rgba(120,120,120,.22); border-radius: 14px; padding: 14px 16px; min-height: 85px;}
.small-muted {opacity:.67; font-size:.9rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(120,120,120,.20); padding: 16px; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)


def normalize_phone(value):
    if value is None or pd.isna(value):
        return ""
    s = re.sub(r"\D", "", str(value))
    if not s:
        return ""
    if s.startswith("00"):
        s = s[2:]
    return s


def phone_variants(phone):
    p = normalize_phone(phone)
    if not p:
        return set()
    variants = {p}
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
        "phone number", "mobile number", "number", "evcontactprimarysale",
        "evcontactsecsale",
    ]
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for p in preferred:
        if p in normalized:
            return normalized[p]

    best_col, best_score = None, 0
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(250)
        if len(sample) == 0:
            continue
        valid = 0
        for v in sample:
            digits = re.sub(r"\D", "", v)
            valid += 7 <= len(digits) <= 16
        score = valid / max(len(sample), 1)
        if score > best_score:
            best_col, best_score = col, score
    return best_col if best_score >= 0.35 else None


def extract_phone_candidates(text):
    raw = re.findall(r"(?:\+?\d[\d\s()./-]{5,}\d)", text or "")
    seen, result = set(), []
    for item in raw:
        p = normalize_phone(item)
        if 7 <= len(p) <= 16 and p not in seen:
            seen.add(p)
            result.append(p)
    return result


def extract_selectable_text_by_page(pdf_bytes, progress_callback=None):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    records = []
    total = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        for p in extract_phone_candidates(text):
            records.append({"PDF Phone": p, "Page": i, "Read Confidence": "High", "Read Method": "PDF Text"})
        if progress_callback:
            progress_callback(i, total)
    return records, total


def ocr_pdf_by_page(pdf_bytes, progress_callback=None, dpi=200):
    if fitz is None or pytesseract is None or Image is None:
        raise RuntimeError("OCR components are not installed. Check requirements.txt and packages.txt.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    records = []
    seen = set()
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(total):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, config="--psm 6")
        for p in extract_phone_candidates(text):
            key = (page_index + 1, p)
            if key not in seen:
                seen.add(key)
                records.append({
                    "PDF Phone": p,
                    "Page": page_index + 1,
                    "Read Confidence": "Medium",
                    "Read Method": "Free OCR",
                })
        if progress_callback:
            progress_callback(page_index + 1, total)
    doc.close()
    return records, total


def smart_extract(pdf_bytes, progress_callback=None):
    """Use free selectable-text extraction first; OCR only pages that look text-empty."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    records = []
    ocr_pages = []
    seen = set()

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        phones = extract_phone_candidates(text)
        for p in phones:
            key = (i, p)
            if key not in seen:
                seen.add(key)
                records.append({"PDF Phone": p, "Page": i, "Read Confidence": "High", "Read Method": "PDF Text"})
        # OCR pages with very little usable text or no detected phone candidate.
        if len(text.strip()) < 40 or not phones:
            ocr_pages.append(i - 1)
        if progress_callback:
            progress_callback("text", i, total)

    if ocr_pages:
        if fitz is None or pytesseract is None or Image is None:
            return records, total, len(ocr_pages)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        matrix = fitz.Matrix(200 / 72.0, 200 / 72.0)
        for n, page_index in enumerate(ocr_pages, start=1):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(image, config="--psm 6")
            for p in extract_phone_candidates(text):
                key = (page_index + 1, p)
                if key not in seen:
                    seen.add(key)
                    records.append({
                        "PDF Phone": p,
                        "Page": page_index + 1,
                        "Read Confidence": "Medium",
                        "Read Method": "Free OCR",
                    })
            if progress_callback:
                progress_callback("ocr", n, len(ocr_pages))
        doc.close()
    return records, total, len(ocr_pages)


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
    export_df["Match_Status"] = "Not Matched"
    export_df["Match_Confidence"] = ""
    export_df["Source_Phone"] = ""
    export_df["Source_Page"] = ""
    export_df["Read_Method"] = ""

    for ref_idx, result_rows in matched_reference_rows.items():
        best = sorted(result_rows, key=lambda r: r["Match Score"], reverse=True)[0]
        export_df.loc[ref_idx, "Match_Status"] = "Matched"
        export_df.loc[ref_idx, "Match_Confidence"] = best["Confidence"]
        export_df.loc[ref_idx, "Source_Phone"] = best["PDF Phone"]
        export_df.loc[ref_idx, "Source_Page"] = best.get("Page", "")
        export_df.loc[ref_idx, "Read_Method"] = best.get("Read Method", "")

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Matched Data")
        results_df.to_excel(writer, index=False, sheet_name="Audit")

    output.seek(0)
    wb = load_workbook(output)
    ws = wb["Matched Data"]
    yellow = PatternFill("solid", fgColor="FFF2CC")
    orange = PatternFill("solid", fgColor="FCE4D6")
    green = PatternFill("solid", fgColor="E2F0D9")
    gray = PatternFill("solid", fgColor="E7E6E6")
    headers = {cell.value: cell.column for cell in ws[1]}
    conf_col = headers.get("Match_Confidence")
    status_col = headers.get("Match_Status")

    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="center")

    for row in range(2, ws.max_row + 1):
        status = ws.cell(row, status_col).value if status_col else ""
        conf = ws.cell(row, conf_col).value if conf_col else ""
        if status == "Matched" and conf == "High":
            fill = green
        elif status == "Matched" and conf == "Medium":
            fill = yellow
        elif status == "Matched" and conf == "Low":
            fill = orange
        else:
            fill = gray
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


st.markdown("""
<div class="hero">
  <h1>Document Matching & Reconciliation</h1>
  <p>Free monthly PDF-to-Excel phone matching with local text extraction, OCR fallback, review controls and an auditable Excel output.</p>
</div>
""", unsafe_allow_html=True)

steps = st.columns(4)
for c, title, desc in zip(
    steps,
    ["1 · Upload", "2 · Analyse", "3 · Review", "4 · Export"],
    ["PDF + Excel", "Free text/OCR extraction", "Confidence & exceptions", "Highlighted Excel"],
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
    phone_col = st.selectbox(
        "Excel phone-number column",
        options=list(df.columns),
        index=list(df.columns).index(phone_col) if phone_col in df.columns else 0,
    )
    with st.expander("Preview selected phone column"):
        preview = df[[phone_col]].dropna().head(12).copy()
        st.dataframe(preview, use_container_width=True, hide_index=True)

analysis_mode = st.radio(
    "Processing method",
    [
        "Smart Free Mode — PDF text + OCR fallback (recommended)",
        "PDF text only — fastest",
        "Free OCR all pages — for scanned PDFs",
    ],
    horizontal=False,
)
st.caption("No Claude, no API key and no paid AI service is used by this app. OCR runs with open-source Tesseract on the Streamlit server.")

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
        if analysis_mode.startswith("Smart"):
            def smart_progress(stage, current, total):
                if stage == "text":
                    pct = 5 + int(current / max(total, 1) * 35)
                    progress.progress(min(pct, 40), text=f"Reading selectable PDF text: page {current}/{total}…")
                else:
                    pct = 40 + int(current / max(total, 1) * 25)
                    progress.progress(min(pct, 65), text=f"Running free OCR fallback: page {current}/{total}…")
            source_records, page_count, ocr_pages = smart_extract(pdf_bytes, smart_progress)
            st.session_state["ocr_pages"] = ocr_pages

        elif analysis_mode.startswith("PDF text"):
            def text_progress(current, total):
                pct = 5 + int(current / max(total, 1) * 50)
                progress.progress(min(pct, 55), text=f"Reading PDF text: page {current}/{total}…")
            source_records, page_count = extract_selectable_text_by_page(pdf_bytes, text_progress)
            st.session_state["ocr_pages"] = 0

        else:
            def ocr_progress(current, total):
                pct = 5 + int(current / max(total, 1) * 60)
                progress.progress(min(pct, 65), text=f"Running free OCR: page {current}/{total}…")
            source_records, page_count = ocr_pdf_by_page(pdf_bytes, ocr_progress)
            st.session_state["ocr_pages"] = page_count

        progress.progress(70, text="Matching phone numbers against Excel…")
        result_rows = []
        matched_reference_rows = {}
        for rec in source_records:
            ref_idx, confidence, score = find_best_match(rec["PDF Phone"], reference_index)
            matched = ref_idx is not None
            row = {
                "PDF Phone": rec["PDF Phone"],
                "Page": rec.get("Page"),
                "Read Method": rec.get("Read Method", ""),
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

        results = pd.DataFrame(result_rows, columns=[
            "PDF Phone", "Page", "Read Method", "Read Confidence", "Status",
            "Confidence", "Match Score", "Excel Row", "Excel Phone"
        ])
        progress.progress(88, text="Preparing highlighted Excel report…")
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
        st.warning("No phone numbers were detected. Try ‘Free OCR all pages’ if the PDF is scanned. Handwriting may still require manual review.")
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
            file_name=f"Matched_Report_{stamp}.xlsx",
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

The workbook includes the highlighted reference data plus a separate **Audit** sheet. Free OCR is intended mainly for printed/scanned numbers; handwriting can be less accurate and should be reviewed.
        """)

st.divider()
st.caption("Free local extraction/OCR supports review; final business decisions remain with the responsible reviewer.")
