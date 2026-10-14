import pymupdf

from app.rag.pdf_processor import chunk_pages, clean_text, extract_pages

PAGES = [
    "Monetary policy. The repo rate was held steady while liquidity conditions were managed carefully "
    "through variable rate operations across the financial year under review by the Reserve Bank.",
    "Inflation. Consumer prices moderated through the second half as food prices eased, supported by "
    "supply side measures and a good monsoon, according to the assessment in this chapter of the report.",
    "Banking. Gross non performing assets of scheduled commercial banks declined to multi year lows, "
    "reflecting stronger balance sheets, better provisioning and improved credit quality overall.",
]


def make_pdf(path):
    doc = pymupdf.open()
    for text in PAGES:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 700), text, fontsize=11)
    doc.save(path)
    doc.close()


def test_pages_extracted_with_exact_numbers(tmp_path):
    pdf = tmp_path / "r.pdf"
    make_pdf(pdf)
    pages = extract_pages(pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "repo rate" in pages[0].text and "Inflation" in pages[1].text


def test_chunks_never_span_pages_and_keep_page_numbers(tmp_path):
    pdf = tmp_path / "r.pdf"
    make_pdf(pdf)
    chunks = chunk_pages(extract_pages(pdf))
    assert chunks and [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        if "repo rate" in c.content:
            assert c.page_number == 1
        if "non performing" in c.content:
            assert c.page_number == 3


def test_clean_text_strips_nul_and_rejoins_hyphenation():
    assert "\x00" not in clean_text("ab\x00cd")
    assert clean_text("with-\ndrawal  of   x") == "withdrawal of x"
