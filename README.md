# Data-Story-Reading-App

A Streamlit-based educational tool that analyzes written data stories by classifying sentences as **"Show"** (descriptive observations) or **"Tell"** (interpretive claims). Built to help students improve their data storytelling skills through AI-powered feedback.

## Features

- **Sentence Classification** — Classifies each sentence using a pre-trained Logistic Regression model with TF-IDF vectorization
- **AI-Generated Explanations** — Provides contextual justifications for each classification via OpenRouter API
- **Key Phrase Highlighting** — Uses sentence-transformers (All-MiniLM-L6-v2) to highlight relevant Show/Tell indicator phrases
- **Student Feedback Collection** — Students can agree/disagree with classifications and leave reflections
- **Visual Breakdown** — Matplotlib charts showing Show vs Tell distribution
- **Email Feedback** — Sends classification summaries to students via Gmail SMTP
- **Database Persistence** — Stores submissions, sentence-level data, and feedback in MySQL
- **Admin Controls** — Week management and authentication for course progression
- **Auto Week & Image Scheduling** — Week number and chart image update automatically based on a configured course start date

## Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | Streamlit |
| ML Classification | scikit-learn (Logistic Regression) |
| NLP Tokenization | NLTK |
| Semantic Embeddings | sentence-transformers |
| Sentence Splitting | OpenAI GPT-4 mini |
| Explanations | OpenRouter API |
| Database | MySQL |
| Visualization | Matplotlib |

## Project Structure

```
Data-Story-Reading-App/
├── streamlit_predict_app.py    # Main application
├── requirements.txt            # Python dependencies
├── Procfile.txt                # Heroku deployment config
├── LICENSE                     # MIT License
├── .streamlit/
│   └── secrets.toml            # API keys & DB credentials (not committed)
├── models/
│   ├── LogisticRegression_All_shots_data_model.pkl
│   └── LogisticRegression_All_shots_data_vectorizer.pkl
├── utils/
│   ├── func.py                 # Helper functions (embeddings, LLM calls, indicators)
│   └── __init__.py
└── images/                     # Chart prompts for student exercises
    ├── dog_walk.png
    ├── chart_prompt.png
    ├── math_reading.png
    ├── stem_prompt.png
    ├── time_survey.png
    ├── wealth_survey.png
    ├── youtube_prompt.png
    └── AI_related_roles.png
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

Create `.streamlit/secrets.toml` with the following:

```toml
OPENAI_API_KEY = "your-openai-api-key"
OPENROUTER_API_KEY = "your-openrouter-api-key"
EMAIL_ADDRESS = "your-gmail@gmail.com"
EMAIL_PASSWORD = "your-gmail-app-password"
DB_HOST = "your-db-host"
DB_PORT = "3306"
DB_NAME = "your-db-name"
DB_USER = "your-db-user"
DB_PASSWORD = "your-db-password"
ADMIN_KEY = "your-admin-password"

# Auto week/image config
COURSE_START_DATE = "YYYY-MM-DD"   # First day of Week 1 in your course
WEEK_IMAGES = [
  "chart_prompt.png",   # Week 1
  "math_reading.png",   # Week 2
  "stem_prompt.png",    # Week 3
  "time_survey.png",    # Week 4
  "wealth_survey.png",  # Week 5
  "youtube_prompt.png", # Week 6
  "AI_related_roles.png", # Week 7
  "dog_walk.png",       # Week 8
]
```

> **Note:** `CURRENT_WEEK` is no longer needed. The app calculates the current week automatically from `COURSE_START_DATE` using `(today − start_date) / 7 + 1`. The chart image is picked from `WEEK_IMAGES` by week index, so both update on their own — no manual changes required each week.

#### Changing the week image schedule

Edit `WEEK_IMAGES` in `secrets.toml` (or Heroku config vars) to assign a different chart image to each week. Filenames refer to images in the `images/` directory. You can repeat filenames across weeks if needed.

### 3. Run the app

```bash
streamlit run streamlit_predict_app.py
```

The app will be available at `http://localhost:8501`.

## Deployment

Configured for Heroku via `Procfile.txt`:

```
web: streamlit run streamlit_predict_app.py --server.port $PORT --server.address 0.0.0.0
```

Set the secrets as Heroku config vars instead of using `secrets.toml`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│              (streamlit_predict_app.py)                         │
│                                                                 │
│   Page 1: Input  →  Page 2: Analysis  →  Page 3: Submission    │
└──────┬──────────────────┬───────────────────────┬───────────────┘
       │                  │                       │
       ▼                  ▼                       ▼
┌─────────────┐  ┌────────────────────┐  ┌────────────────┐
│   NLTK      │  │  ML Pipeline       │  │  MySQL         │
│  Tokenizer  │  │                    │  │  Database      │
│             │  │  TF-IDF Vectorizer │  │                │
│  OpenAI API │  │  + LogisticRegr.   │  │  students      │
│  (GPT-4.1   │  │  (models/*.pkl)    │  │  weeks         │
│   mini)     │  │                    │  │  student_inputs │
│  Sentence   │  │  sentence-         │  │  student_      │
│  Splitting  │  │  transformers      │  │   sentences    │
└─────────────┘  │  (highlighting)    │  └───────┬────────┘
                 │                    │          │
                 │  OpenRouter API    │          ▼
                 │  (GPT-4o-mini)     │  ┌────────────────┐
                 │  Explanations      │  │  Gmail SMTP    │
                 └────────────────────┘  │  Email Feedback│
                                         └────────────────┘
```

### Data Flow

1. **Input** — Student enters name, email, story title, and writes a data story about a provided chart
2. **Tokenization** — NLTK splits text into sentences. Long/complex sentences are further split using **OpenAI API** (GPT-4.1-mini)
3. **Classification** — Each sentence is vectorized (TF-IDF) and classified as Show/Tell by the **local Logistic Regression model**
4. **Enrichment** — **sentence-transformers** highlights key phrases; **OpenRouter API** (GPT-4o-mini) generates explanations
5. **Feedback** — Student reviews classifications, agrees/disagrees, and writes a reflection
6. **Persistence** — All data is saved to **MySQL**; a summary email is sent via **Gmail SMTP**

### Database Schema

```mermaid
erDiagram
    students ||--o{ student_inputs : "submits"
    weeks ||--o{ student_inputs : "contains"
    weeks ||--o{ student_sentences : "tracks"
    student_inputs ||--|{ student_sentences : "comprises"

    students {
        int student_id PK
        string full_name
        string email UK
    }

    weeks {
        int week_id PK
        int week_number UK
        string label
    }

    student_inputs {
        int input_id PK
        int student_id FK
        int week_id FK
        string student_name
        string email
        string title
        text story
        int total_sentences
        int show_sentences
        int tell_sentences
        text reflection
        string helpfulness
        text comments
    }

    student_sentences {
        int sentence_id PK
        int input_id FK
        int week_id FK
        int sentence_idx
        text sentence_text
        string model_label
        boolean student_agree
        string highlight_words
        text explanation
    }
```

### External API Usage

| API | Model | Purpose | Called In |
|-----|-------|---------|----------|
| **OpenAI** | GPT-4.1-mini | Splits complex sentences into individual ones | `split_with_llm()` in `streamlit_predict_app.py` |
| **OpenRouter** | GPT-4o-mini | Generates 1-2 sentence explanations for each classification | `call_openrouter_llm()` in `utils/func.py` |

## How It Works

1. **Input** — Students enter their name, email, story title, and write a data story about a provided chart
2. **Analysis** — Sentences are tokenized (NLTK), optionally split with GPT-4.1-mini, then classified by the ML model. Each sentence gets an AI explanation and highlighted key phrases
3. **Feedback** — Students review classifications, agree or disagree, and write a reflection
4. **Submission** — Results are saved to MySQL and a feedback email is sent to the student

## License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.
