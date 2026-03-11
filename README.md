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
CURRENT_WEEK = "5"
```

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

```
students
├── student_id (PK, AUTO_INCREMENT)
├── full_name
└── email (UNIQUE)

weeks
├── week_id (PK, AUTO_INCREMENT)
├── week_number (UNIQUE)
└── label

student_inputs
├── input_id (PK, AUTO_INCREMENT)
├── student_id (FK → students)
├── week_id (FK → weeks)
├── student_name
├── email
├── title
├── story
├── total_sentences
├── show_sentences
├── tell_sentences
├── reflection
├── helpfulness
└── comments

student_sentences
├── sentence_id (PK, AUTO_INCREMENT)
├── input_id (FK → student_inputs)
├── week_id (FK → weeks)
├── sentence_idx
├── sentence_text
├── model_label
├── student_agree
├── highlight_words
└── explanation
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
