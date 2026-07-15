# Resume Screening System

Upload a resume (PDF or DOCX) → get an automatic **job category prediction**
(ML classifier) and, if you paste in a job description, a **match score**
(text similarity).

## How it works (the short version)

```
resume file  ──▶  extract text  ──▶  clean text  ──┬──▶  TF-IDF + Logistic Regression ──▶ predicted category
                                                      │
                       job description (optional) ──┴──▶  TF-IDF cosine similarity ──▶ match score %
```

- **Classification**: a `LogisticRegression` model trained on ~960 labelled
  resumes (25 job categories) using TF-IDF text features. Trained once,
  offline, saved to disk, loaded by the API at startup.
- **Matching**: a lightweight TF-IDF + cosine similarity between the resume
  text and whatever job description you paste in. No training needed for
  this part — it just compares the two documents.

## Project structure

```
resume-screening-system/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app + CORS
│   │   ├── schemas.py             # Pydantic response models
│   │   ├── routers/
│   │   │   └── resume.py          # POST /api/screen
│   │   ├── services/
│   │   │   ├── parser.py          # PDF/DOCX -> text
│   │   │   ├── preprocess.py      # text cleaning (shared w/ training)
│   │   │   ├── classifier.py      # loads model, predicts category
│   │   │   └── matcher.py         # resume vs job-description similarity
│   │   └── ml_models/             # trained model artifacts (.pkl)
│   ├── ml_training/
│   │   ├── train_model.py         # run this to (re)train the model
│   │   └── data/
│   │       └── UpdatedResumeDataSet.csv
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## Setup

### 1. Backend

cd backend
pip install -r requirements.txt
```

```bash
python ml_training/train_model.py
```

Start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs (FastAPI
generates this automatically from the Pydantic schemas).

### 2. Frontend

No build step — just open the file, or serve it so it isn't a `file://`
origin (nicer for the browser's dev tools / CORS):

```bash
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500`. Make sure the backend is running on
port 8000 first — `script.js` points at `http://localhost:8000`.

## API

`POST /api/screen`

| field             | type | required | notes                               |
|-------------------|------|----------|--------------------------------------|
| `file`            | file | yes      | `.pdf` or `.docx`, max 5MB           |
| `job_description` | text | no       | if given, also returns a match score |

Response:

```json
{
  "filename": "resume.pdf",
  "predicted_category": "Data Science",
  "confidence": 0.87,
  "top_predictions": [
    {"category": "Data Science", "probability": 0.87},
    {"category": "Python Developer", "probability": 0.06},
    {"category": "Hadoop", "probability": 0.03}
  ],
  "match_score": 62.4,
  "matched_keywords": ["python", "sql", "machine", "learning"],
  "extracted_text_preview": "..."
}
```

## Notes / things you could improve next

- **Dataset reality check**: the raw CSV has 962 rows but 796 of them are
  exact-duplicate resumes — only **166 are actually unique**. The original
  training script did a random train/test split on the raw (undeduplicated)
  data, so duplicate copies of the same resume ended up in both train and
  test, which is why it reported ~99.5% accuracy. That number was not real.
  `train_model.py` now de-duplicates first and evaluates with stratified
  k-fold cross-validation instead, which gives an honest estimate: **~89%
  accuracy / ~0.86 macro-F1** on genuinely unseen resumes. That's the number
  to trust, and it's actually a solid result for 166 examples across 25
  classes — but it also means confidence scores on real-world resumes will
  be more modest (and more honest) than before.
- **To meaningfully improve accuracy further, you need more unique labeled
  resumes** — no amount of algorithm tuning fixes a 166-example dataset.
  Concrete ways to grow it, roughly in order of effort:
  1. Merge in another public labeled resume dataset (search Kaggle/Hugging
     Face for "resume classification dataset" — several exist with
     different categories; you'll need to reconcile category names).
  2. Reduce from 25 categories to a smaller set of broader roles if some
     categories aren't important to your use case — fewer classes means
     each one needs less data to be reliable.
  3. Collect real resumes from your own hiring pipeline (with consent) and
     label them — even 20-30 more per category would help a lot given how
     small this is.
  4. As a stopgap, LLM-generated synthetic resumes per category can pad out
     the smallest classes, but they tend to be more homogeneous than real
     resumes, so treat them as a supplement, not a replacement.
- The matcher fits a fresh TF-IDF per request on just the resume + job
  description; that's fine for one-off comparisons but wouldn't scale well
  to ranking hundreds of resumes against one job description efficiently —
  you'd want to vectorize once and batch-compare for that.
- No database yet — nothing is persisted. Adding one (e.g. SQLite via
  SQLAlchemy) would be a natural next step if you want a history of past
  screenings.
- No auth — anyone who can reach the API can use it. Fine for local/demo
  use, not for a public deployment.
