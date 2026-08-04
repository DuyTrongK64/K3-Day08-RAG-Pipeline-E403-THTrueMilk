"""Task 3 - Convert files in data/landing to Markdown with MarkItDown."""

import json
from pathlib import Path

from markitdown import MarkItDown


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX files and preserve the legal subdirectory."""
    input_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()
    outputs = []

    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".doc", ".docx"}
    )
    if not files:
        raise FileNotFoundError(f"Không có PDF/DOC/DOCX trong {input_dir}")

    for filepath in files:
        print(f"Converting legal: {filepath.name}")
        result = converter.convert(str(filepath))
        extracted_text = (result.text_content or "").strip()

        # MarkItDown does not perform OCR. Signed/scanned PDFs can therefore
        # contain very little extractable text; still write the conversion and
        # report it clearly instead of silently treating it as rich text.
        if len(extracted_text) < 200:
            print(
                f"  WARNING: only {len(extracted_text)} characters extracted; "
                "the PDF may be scanned."
            )

        header = (
            "---\n"
            f"source_file: {json.dumps(filepath.name, ensure_ascii=False)}\n"
            "document_type: \"legal_document\"\n"
            "language: \"vi\"\n"
            "conversion_tool: \"Microsoft MarkItDown\"\n"
            "---\n\n"
            f"# {filepath.stem}\n\n"
        )
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + extracted_text + "\n", encoding="utf-8")
        outputs.append(output_path)
        print(f"  Saved: {output_path}")

    return outputs


def convert_news_articles() -> list[Path]:
    """Extract content_markdown and metadata from Task 2 JSON files."""
    input_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Không có JSON trong {input_dir}")

    for filepath in files:
        print(f"Converting news: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        content_markdown = str(data.get("content_markdown", "")).strip()
        if not content_markdown:
            raise ValueError(f"{filepath.name} thiếu content_markdown")

        metadata_keys = (
            "url",
            "author",
            "published_date",
            "date_crawled",
            "source_domain",
            "source_owner",
            "topic",
            "document_type",
            "language",
        )
        header_lines = ["---"]
        for key in metadata_keys:
            value = data.get(key)
            if value is not None:
                header_lines.append(
                    f"{key}: {json.dumps(str(value), ensure_ascii=False)}"
                )
        header_lines.extend(
            [
                f"source_file: {json.dumps(filepath.name, ensure_ascii=False)}",
                "---",
                "",
                f"# {data.get('title', filepath.stem)}",
                "",
            ]
        )

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(
            "\n".join(header_lines) + content_markdown + "\n",
            encoding="utf-8",
        )
        outputs.append(output_path)
        print(f"  Saved: {output_path}")

    return outputs


def convert_all() -> list[Path]:
    """Convert all supported landing files while keeping legal/news layout."""
    print("=" * 60)
    print("Task 3: Convert to Markdown with Microsoft MarkItDown")
    print("=" * 60)
    outputs = convert_legal_docs() + convert_news_articles()
    print(f"Done: {len(outputs)} Markdown files in {OUTPUT_DIR}")
    return outputs


if __name__ == "__main__":
    convert_all()
