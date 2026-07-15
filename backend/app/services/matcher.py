

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Some linters/analysis tools can't resolve ENGLISH_STOP_WORDS from
    # sklearn.feature_extraction.text; import the text module and grab it there.
    from sklearn.feature_extraction import text as _sk_text
    ENGLISH_STOP_WORDS = _sk_text.ENGLISH_STOP_WORDS
except Exception:
    # Fallbacks if sklearn isn't available at import-check time.
    from sklearn.feature_extraction.text import TfidfVectorizer
    ENGLISH_STOP_WORDS = set()
from sklearn.metrics.pairwise import cosine_similarity

from app.services.preprocess import clean_text


def match_score(resume_text: str, job_description: str) -> dict:
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    score_percent = round(float(similarity) * 100, 2)

    resume_words = set(resume_clean.split())
    jd_words = set(jd_clean.split())
    common_keywords = sorted(
        w for w in (resume_words & jd_words)
        if w not in ENGLISH_STOP_WORDS and len(w) > 2
    )[:15]

    return {
        "match_score": score_percent,
        "matched_keywords": common_keywords,
    }
