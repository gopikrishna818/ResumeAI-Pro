"""
model.py - Core NLP engine for resume screening
Uses TF-IDF vectorization + cosine similarity for candidate ranking
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load spaCy model for NER (skill extraction)
# Run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    print("⚠  spaCy model not found. Run: python -m spacy download en_core_web_sm")


# ── Common tech skill keywords for keyword-based matching ──────────────────
TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "r", "sql", "nosql", "bash", "shell", "html", "css", "react", "vue",
    "angular", "node.js", "fastapi", "flask", "django", "spring", "aws",
    "azure", "gcp", "docker", "kubernetes", "terraform", "git", "linux",
    "machine learning", "deep learning", "nlp", "computer vision", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "spark",
    "hadoop", "kafka", "airflow", "dbt", "postgresql", "mysql", "mongodb",
    "redis", "elasticsearch", "tableau", "power bi", "excel", "agile",
    "scrum", "rest", "graphql", "microservices", "ci/cd", "devops",
}


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)      # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


def extract_skills(text: str) -> set[str]:
    """Extract skills from text using keyword matching + NER."""
    cleaned = text.lower()
    found = set()

    # Keyword matching
    for skill in TECH_SKILLS:
        if skill in cleaned:
            found.add(skill)

    # spaCy NER for extra entities (ORG often catches tech names)
    if nlp:
        doc = nlp(text[:5000])   # limit for speed
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT") and len(ent.text) > 2:
                found.add(ent.text.lower())

    return found


def compute_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute TF-IDF cosine similarity between one resume and the JD.
    Returns a float in [0, 1].
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=5000,
    )
    corpus = [preprocess(resume_text), preprocess(jd_text)]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return float(score)


def rank_candidates(resumes: list[dict], jd_text: str) -> list[dict]:
    """
    Rank a list of resumes against a job description.

    Args:
        resumes: list of {"name": str, "text": str}
        jd_text: raw job description string

    Returns:
        Sorted list of result dicts with score, matched/missing skills, rank.
    """
    jd_skills = extract_skills(jd_text)
    results = []

    for resume in resumes:
        text = resume["text"]
        sim = compute_similarity(text, jd_text)
        score = round(sim * 100)

        resume_skills = extract_skills(text)
        matched = sorted(resume_skills & jd_skills)
        missing = sorted(jd_skills - resume_skills)

        results.append({
            "name": resume["name"],
            "score": score,
            "matched_skills": matched,
            "missing_skills": missing,
            "similarity_raw": round(sim, 4),
        })

    # Sort by score descending, assign rank
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def batch_rank(resumes: list[dict], jd_text: str) -> list[dict]:
    """
    Batch TF-IDF ranking — fits one vectorizer on ALL documents at once
    for more consistent IDF weights across candidates.
    """
    jd_skills = extract_skills(jd_text)
    all_texts = [preprocess(r["text"]) for r in resumes] + [preprocess(jd_text)]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
    )
    tfidf = vectorizer.fit_transform(all_texts)
    jd_vec = tfidf[-1]      # last entry is the JD

    results = []
    for i, resume in enumerate(resumes):
        sim = cosine_similarity(tfidf[i], jd_vec)[0][0]
        score = round(sim * 100)
        resume_skills = extract_skills(resume["text"])
        results.append({
            "name": resume["name"],
            "score": score,
            "matched_skills": sorted(resume_skills & jd_skills),
            "missing_skills": sorted(jd_skills - resume_skills),
            "similarity_raw": round(float(sim), 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results
