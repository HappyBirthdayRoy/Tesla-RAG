from __future__ import annotations

import argparse

from tesla_rag.config import DEFAULT_CHROMA_DIR, configured_pdf_paths, validate_pdf_paths
from tesla_rag.service import RagService


def run_ingest(chroma_dir: str, pdf_paths: list[str]) -> int:
    validated = validate_pdf_paths(pdf_paths)
    service = RagService(persist_dir=chroma_dir)
    inserted = service.ingest(validated)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Tesla PDFs into Chroma")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--pdf", action="append", default=None, help="Explicit PDF path; can be used multiple times")
    args = parser.parse_args()

    pdf_paths = args.pdf if args.pdf else configured_pdf_paths()
    inserted = run_ingest(chroma_dir=args.chroma_dir, pdf_paths=pdf_paths)
    print(f"Inserted chunks: {inserted}")


if __name__ == "__main__":
    main()
