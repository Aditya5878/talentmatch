import hashlib
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument


def extract_text(file_path: str) -> str:
    """Extract plain text from a file based on its extension.

    Supports PDF (via pypdf), DOCX/DOC (via python-docx), and plain text files.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        Extracted text content as a single string.

    Raises:
        ValueError: If the file format is not supported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(path)
    else:
        return _extract_txt(path)


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf.

    Args:
        path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx.

    Args:
        path: Path to the DOCX file.

    Returns:
        Concatenated text from all paragraphs.
    """
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_txt(path: Path) -> str:
    """Read a plain text file.

    Args:
        path: Path to the text file.

    Returns:
        File content as a string.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute a SHA-256 hash of the given file bytes.

    Used for idempotency — re-ingesting the same file detects duplicates
    by comparing this hash against stored records.

    Args:
        file_bytes: Raw file content.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(file_bytes).hexdigest()
