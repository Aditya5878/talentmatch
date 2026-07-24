from typing import Any


def chunk_resume(parsed: dict[str, Any]) -> list[dict[str, str]]:
    """Chunk a parsed resume into section-based text segments.

    Creates separate chunks for skills (single chunk), experience entries
    (one chunk per job), and education entries (one chunk per degree).
    Section-based chunking retrieves better than fixed-length chunking
    because each chunk represents a coherent semantic unit.

    Args:
        parsed: Parsed resume data from parse_resume().

    Returns:
        List of dicts with 'section' (str) and 'text' (str) keys.
    """
    chunks: list[dict[str, str]] = []

    skills = parsed.get("skills", [])
    if skills:
        chunks.append({
            "section": "skills",
            "text": "Skills: " + ", ".join(skills),
        })

    for exp in parsed.get("experience", []):
        title = exp.get("title", "")
        company = exp.get("company", "")
        desc = exp.get("description", "")
        text = f"{title} at {company}: {desc}" if title and company else f"{title} at {company}{desc}"
        chunks.append({"section": "experience", "text": text.strip()})

    for edu in parsed.get("education", []):
        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        text = f"{degree} - {institution}" if degree and institution else f"{degree} {institution}"
        chunks.append({"section": "education", "text": text.strip()})

    return chunks


def chunk_jd(parsed: dict[str, Any]) -> list[dict[str, str]]:
    """Chunk a parsed job description into section-based text segments.

    Creates separate chunks for required skills, nice-to-have skills,
    and individual responsibilities.

    Args:
        parsed: Parsed JD data from parse_jd().

    Returns:
        List of dicts with 'section' (str) and 'text' (str) keys.
    """
    chunks: list[dict[str, str]] = []

    required = parsed.get("required_skills", [])
    if required:
        chunks.append({
            "section": "required_skills",
            "text": "Required skills: " + ", ".join(required),
        })

    nice_to_have = parsed.get("nice_to_have", [])
    if nice_to_have:
        chunks.append({
            "section": "nice_to_have",
            "text": "Nice to have: " + ", ".join(nice_to_have),
        })

    for resp in parsed.get("responsibilities", []):
        chunks.append({"section": "responsibilities", "text": resp})

    return chunks
