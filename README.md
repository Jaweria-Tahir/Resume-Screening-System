# AI RESUME SCANNER

Upload a resume (PDF/DOCX) → get an ML-predicted job category, and, if you paste in a job description, an instant match score.
<img width="1900" height="860" alt="image" src="https://github.com/user-attachments/assets/8d00fcd8-dafa-4f12-87ad-02702a850401" />
<img width="1902" height="832" alt="image" src="https://github.com/user-attachments/assets/4f4f9782-cff3-47b5-9a7c-9d2c438e4baf" />
<img width="1871" height="840" alt="image" src="https://github.com/user-attachments/assets/7e87b93b-6fe7-4a9e-9bb8-4d1aa4678850" />

Live Demo:  https://ai-resume-scanner-lake.vercel.app/


This is a small end-to-end ML application: a resume comes in as a file, a TF-IDF + Logistic Regression classifier predicts which of 25 job categories it belongs to, and — if a job description is pasted in — a TF-IDF cosine-similarity matcher scores how well the resume fits that specific role.It's built as a realistic, minimal-scope portfolio project: a FastAPI backend serving a scikit-learn model, a static frontend, and a documented, honest evaluation of the model's limits

## Features
Resume upload — accepts .pdf and .docx, up to 5MB
Job category prediction — 25 categories, with confidence score and top-3 alternatives
Job-match scoring — optional job description input returns a similarity % and matched keywords
Input validation — rejects unreadable, empty, or non-resume-looking files with clear error messages
Auto-generated API docs — interactive Swagger UI via FastAPI

## Tech Stack

| Layer                | Technology                                                    |
| -------------------- | ------------------------------------------------------------- |
| **API**              | FastAPI, Uvicorn                                              |
| **Machine Learning** | scikit-learn (TF-IDF, Logistic Regression, Cosine Similarity) |
| **File Parsing**     | pdfplumber (PDF), python-docx (DOCX)                          |
| **Frontend**         | HTML, CSS, Vanilla JavaScript                                 |
| **Deployment**       | Dockerized FastAPI on Render, Frontend on Vercel              |

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
```bash
cd backend
pip install -r requirements.txt

# (Optional) Retrain the model from scratch
python ml_training/train_model.py

# Start the API
uvicorn app.main:app --reload --port 8000
```
### 2. Frontend
```bash
cd frontend
python -m http.server 5500
```
## Example response
```json
{
  "filename": "resume.pdf",
  "predicted_category": "Data Science",
  "confidence": 0.87,
  "top_predictions": [
    {
      "category": "Data Science",
      "probability": 0.87
    },
    {
      "category": "Python Developer",
      "probability": 0.06
    },
    {
      "category": "Hadoop",
      "probability": 0.03
    }
  ],
  "match_score": 62.4,
  "matched_keywords": [
    "python",
    "sql",
    "machine",
    "learning"
  ],
  "extracted_text_preview": "Experienced data scientist with expertise in Python, SQL, machine learning, and data visualization..."
}
```

## Deployment
Backend: Dockerized FastAPI app, deployed via Render (render.yaml included)
Frontend: Static site deployed on Vercel.Deployment

## Future Improvements
-Store screening results and resume history using a database.
-Add user authentication and role-based access control.
-Improve the resume matching algorithm using semantic similarity instead of keyword matching.
-Optimize the application's performance to reduce prediction time.
-Enhance the UI/UX and add progress indicators during resume analysis.

Backend: Dockerized FastAPI app, deployed via Render (render.yaml included).
Frontend: Static site deployed on Vercel.
