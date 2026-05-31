"""Chunk and prepare the Scotia T&C PDF for Vertex AI Search indexing.

Usage
-----
    python knowledge_base/ingest.py --pdf knowledge_base/scotia_tnc.pdf
    python knowledge_base/ingest.py --pdf knowledge_base/scotia_tnc.pdf --verify
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pypdf


logger = logging.getLogger(__name__)

CHUNK_SIZE = 512       # tokens (approximated as words here)
CHUNK_OVERLAP = 64
SOURCE_NAME = "scotia_tnc_v1"

# Pattern to detect section headings like "Section 4.2" or "4.2 Billing"
_SECTION_RE = re.compile(r"(section\s+\d[\d.]*|^\d[\d.]+\s+\w)", re.IGNORECASE | re.MULTILINE)


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    list[dict]
        List of page dicts with ``page_number`` and ``text``.
    """
    pages = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_number": i, "text": text.strip()})
    logger.info("Extracted text from %d pages", len(pages))
    return pages


def _detect_section(text: str) -> str:
    """Extract the first section heading found in a text chunk."""
    match = _SECTION_RE.search(text)
    return match.group(0).strip() if match else "Unknown Section"


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split page text into overlapping word-based chunks.

    Parameters
    ----------
    pages : list[dict]
        Output of ``extract_text_from_pdf``.
    chunk_size : int
        Approximate chunk size in words.
    overlap : int
        Word overlap between consecutive chunks.

    Returns
    -------
    list[dict]
        Chunk records with ``clause_text``, ``section``, ``page_number``, ``source``.
    """
    chunks = []
    for page in pages:
        words = page["text"].split()
        if not words:
            continue
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            chunk_words = words[start: start + chunk_size]
            if not chunk_words:
                continue
            clause_text = " ".join(chunk_words)
            chunks.append({
                "clause_text": clause_text,
                "section": _detect_section(clause_text),
                "page_number": page["page_number"],
                "source": SOURCE_NAME,
            })

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks


def save_chunks_jsonl(chunks: list[dict], output_path: Path) -> None:
    """Save chunks as JSONL for Vertex AI Search import.

    Parameters
    ----------
    chunks : list[dict]
        Chunk records.
    output_path : Path
        Destination JSONL file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Saved %d chunks to %s", len(chunks), output_path)


def ingest(pdf_path: str | Path, output_path: str | Path | None = None) -> list[dict]:
    """Extract, chunk, and save T&C PDF content.

    After running this, upload the generated JSONL to Vertex AI Search:
      gcloud discoveryengine datastores documents import \
        --project=PROJECT \
        --location=global \
        --datastore=DATASTORE_ID \
        --source-gcs-uri=gs://BUCKET/tnc_chunks.jsonl

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF.
    output_path : str or Path or None
        Destination JSONL file. Defaults to ``data/processed/tnc_chunks.jsonl``.

    Returns
    -------
    list[dict]
        All generated chunks.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path or "data/processed/tnc_chunks.jsonl")

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    save_chunks_jsonl(chunks, output_path)
    return chunks


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest T&C PDF into Vertex AI Search.")
    parser.add_argument("--pdf", default="knowledge_base/scotia_tnc.pdf", help="Path to PDF.")
    parser.add_argument("--output", default="data/processed/tnc_chunks.jsonl", help="Output JSONL path.")
    parser.add_argument("--verify", action="store_true", help="Verify chunk count and print first chunk.")
    args = parser.parse_args()

    chunks = ingest(args.pdf, args.output)

    if args.verify:
        errors = [c for c in chunks if not c.get("clause_text")]
        print(f"Total chunks: {len(chunks)}, errors: {len(errors)}")
        if chunks:
            print("First chunk:", json.dumps(chunks[0], indent=2))
        sys.exit(1 if errors else 0)
