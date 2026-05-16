"""
utils.py - File I/O and text preprocessing utilities
Handles PDF extraction, text cleaning, and batch loading
"""

import os
import re
import json
from pathlib import Path


# ── PDF Extraction ─────────────────────────────────────────────────────────
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from a PDF file using PyPDF2.
    Falls back gracefully if extraction fails.
    """
    try:
        import PyPDF2
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()
    except ImportError:
        raise ImportError("Install PyPDF2: pip install PyPDF2")
    except Exception as e:
        print(f"⚠  Failed to extract {file_path}: {e}")
        return ""


def extract_text_from_txt(file_path: str, encoding: str = "utf-8") -> str:
    """Read plain text file."""
    try:
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠  Failed to read {file_path}: {e}")
        return ""


def extract_text(file_path: str) -> str:
    """
    Auto-detect file type and extract text.
    Supports .pdf and .txt files.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix in (".txt", ".md", ".text"):
        return extract_text_from_txt(file_path)
    else:
        print(f"⚠  Unsupported file type: {suffix}")
        return ""


# ── Batch Loader ───────────────────────────────────────────────────────────
def load_resumes_from_folder(folder_path: str) -> list[dict]:
    """
    Load all .pdf and .txt resumes from a folder.

    Returns:
        List of {"name": str, "file": str, "text": str}
    """
    folder = Path(folder_path)
    supported = {".pdf", ".txt", ".md"}
    resumes = []

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in supported:
            text = extract_text(str(file))
            if text:
                resumes.append({
                    "name": file.stem,       # filename without extension
                    "file": file.name,
                    "text": text,
                })
                print(f"✅ Loaded: {file.name} ({len(text)} chars)")
            else:
                print(f"⚠  Skipped (empty): {file.name}")

    return resumes


def load_jd_from_file(file_path: str) -> str:
    """Load job description from a text or PDF file."""
    return extract_text(file_path)


# ── Text Utilities ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove extra whitespace, normalize line breaks."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def truncate(text: str, max_chars: int = 5000) -> str:
    """Truncate text to a max character limit for API efficiency."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


# ── Results Export ─────────────────────────────────────────────────────────
def save_results_json(results: list[dict], output_path: str = "results.json") -> None:
    """Save ranking results to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to {output_path}")


def save_results_csv(results: list[dict], output_path: str = "results.csv") -> None:
    """Save ranking results to a CSV file."""
    import csv
    if not results:
        return
    fieldnames = ["rank", "name", "score", "matched_skills", "missing_skills", "similarity_raw"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["matched_skills"] = ", ".join(r.get("matched_skills", []))
            row["missing_skills"] = ", ".join(r.get("missing_skills", []))
            writer.writerow(row)
    print(f"💾 Results saved to {output_path}")


def print_results_table(results: list[dict]) -> None:
    """Pretty-print ranking results to terminal."""
    print("\n" + "=" * 65)
    print(f"{'Rank':<6} {'Candidate':<28} {'Score':>6}  {'Matched Skills'}")
    print("=" * 65)
    for r in results:
        matched = ", ".join(r.get("matched_skills", [])[:4])
        if len(r.get("matched_skills", [])) > 4:
            matched += "…"
        print(f"#{r['rank']:<5} {r['name']:<28} {r['score']:>5}%  {matched}")
    print("=" * 65 + "\n")
