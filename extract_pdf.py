"""Optional step 0: PDF -> markdown (text + tables) with Docling.

    python extract_pdf.py SOPs.pdf data/sop.md

Review the markdown once (headings, table rows) before indexing; the chunker relies on it.
"""
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter

src, dst = sys.argv[1], Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(DocumentConverter().convert(src).document.export_to_markdown(), encoding="utf-8")
print("saved", dst)
