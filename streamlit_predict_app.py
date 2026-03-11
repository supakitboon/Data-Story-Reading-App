import os
import re
import smtplib
import mysql.connector
import joblib
import nltk
import streamlit as st
import matplotlib.pyplot as plt
from email.message import EmailMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

st.set_page_config(initial_sidebar_state="collapsed")

import utils.func as ufunc  # embedding + OpenRouter helpers

# =========================
# NLTK
# =========================
nltk.download("punkt")
try:
    nltk.download("punkt_tab")  # harmless if missing
except:
    pass

# =========================
# Secrets helper
# =========================
def get_secret(name: str):
    try:
        return st.secrets[name]
    except Exception:
        return None


# =========================
# Email secrets
# =========================
EMAIL_ADDRESS = (get_secret("EMAIL_ADDRESS") or "").strip()
EMAIL_PASSWORD = (get_secret("EMAIL_PASSWORD") or "").strip()
if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
    st.error(
        "Email creds missing. Set EMAIL_ADDRESS and EMAIL_PASSWORD "
        "(Gmail App Password) in Streamlit Secrets."
    )
    st.stop()

# =========================
# OpenAI key (for sentence splitting)
# =========================
OPENAI_API_KEY = (get_secret("OPENAI_API_KEY") or "").strip()
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# OpenRouter key (for explanations)
# =========================
OPENROUTER_API_KEY = (get_secret("OPENROUTER_API_KEY") or "").strip()

# =========================
# Model + Vectorizer
# =========================
def load_model_and_vectorizer():
    try:
        model = joblib.load("models/LogisticRegression_All_shots_data_model.pkl")
        vectorizer = joblib.load(
            "models/LogisticRegression_All_shots_data_vectorizer.pkl"
        )
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        st.stop()


def predict_sentences(sentences, model, vectorizer):
    tokens = [" ".join(nltk.word_tokenize(s.lower())) for s in sentences]
    return model.predict(vectorizer.transform(tokens))


@st.cache_resource
def load_embedding_model():
    return ufunc.load_embedding_model()

# =========================
# Highlight helper
# =========================
def highlight_sentence(sentence: str, highlights: list[str]) -> str:
    """
    Wrap each highlight phrase in <b>...</b> inside the sentence.
    Simple literal substring replacement, longest phrases first.
    """
    html = sentence
    for h in sorted(highlights or [], key=len, reverse=True):
        h = h.strip()
        if not h:
            continue
        pattern = re.escape(h)
        html = re.sub(pattern, f"<b><u>{h}</u></b>", html)
    return html

# =========================
# LLM helpers for sentence splitting
# =========================
def looks_like_multi_sentence(text: str) -> bool:
    text = text.strip()
    if not text:
        return False

    if len(text) > 200 and text.count(".") == 0:
        return True
    if (text.count("?") + text.count("!")) >= 2 and text.count(".") == 0:
        return True
    if "\n" in text:
        return True
    if text.count(",") >= 3 and text.count(".") == 0:
        return True
    if text.count("-") >= 2 and text.count(".") == 0:
        return True

    return False


def split_with_llm(text: str) -> list[str]:
    system_msg = (
        "You split text into sentences for data cleaning.\n"
        "- Do NOT remove or change any words, punctuation, or capitalization.\n"
        "- Only decide where one sentence ends and the next begins.\n"
        "- Return the sentences joined by the delimiter: |||\n"
    )

    user_msg = f"Text to split:\n{text}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )

        raw = resp.choices[0].message.content or ""
        parts = [s.strip() for s in raw.split("|||") if s.strip()]
        print(f"⚡ LLM split applied: {text[:80]}...")
        return parts if parts else [text]
    except Exception as e:
        print(f"[LLM ERROR] {e} — returning original sentence.")
        return [text]

# =========================
# Sentence boundary normalizer
# =========================
def normalize_sentence_boundaries(text: str) -> str:
    """
    Ensure there's a space after sentence-ending punctuation like .?! 
    when followed immediately by a letter, e.g. 'go.Red' -> 'go. Red'
    """
    return re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)


def smart_sentence_tokenize(story: str) -> list[str]:
    # 🔧 fix missing spaces after .?! before tokenizing
    story = normalize_sentence_boundaries(story)

    raw_sentences = nltk.sent_tokenize(story)
    final_sentences: list[str] = []
    for s in raw_sentences:
        if looks_like_multi_sentence(s):
            final_sentences.extend(split_with_llm(s))
        else:
            final_sentences.append(s)
    return final_sentences

# =========================
# DB helpers
# =========================
def get_db_connection():
    try:
        return mysql.connector.connect(
            host=get_secret("DB_HOST"),
            port=int(get_secret("DB_PORT")),
            database=get_secret("DB_NAME"),
            user=get_secret("DB_USER"),
            password=get_secret("DB_PASSWORD"),
            autocommit=False,
        )
    except mysql.connector.Error as err:
        st.error(f"Database Connection Error: {err}")
        return None


def get_or_create_student(full_name: str, email: str):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        email_l = (email or "").strip().lower()
        cur.execute(
            """
            INSERT INTO students (full_name, email)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE full_name = VALUES(full_name)
            """,
            (full_name.strip(), email_l),
        )
        if cur.lastrowid:
            student_id = cur.lastrowid
        else:
            cur.execute("SELECT student_id FROM students WHERE email=%s", (email_l,))
            row = cur.fetchone()
            student_id = row[0] if row else None
        conn.commit()
        return student_id
    except mysql.connector.Error as err:
        conn.rollback()
        st.error(f"⚠️ MySQL Error (student): {err}")
        return None
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


def get_or_create_week(week_number: int, label: str | None = None):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        lbl = label if label else f"Week {int(week_number)}"
        cur.execute(
            """
            INSERT INTO weeks (week_number, label)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE label = COALESCE(VALUES(label), label)
            """,
            (int(week_number), lbl),
        )
        if cur.lastrowid:
            week_id = cur.lastrowid
        else:
            cur.execute(
                "SELECT week_id FROM weeks WHERE week_number=%s",
                (int(week_number),),
            )
            row = cur.fetchone()
            week_id = row[0] if row else None
        conn.commit()
        return week_id
    except mysql.connector.Error as err:
        conn.rollback()
        st.error(f"⚠️ MySQL Error (week): {err}")
        return None
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


def has_existing_submission(student_id: int, week_id: int) -> bool:
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM student_inputs WHERE student_id=%s AND week_id=%s LIMIT 1",
            (student_id, week_id),
        )
        return cur.fetchone() is not None
    except mysql.connector.Error as err:
        st.error(f"DB error checking existing submission: {err}")
        return False
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


def insert_submission_and_sentences(
    student_id,
    week_id,
    student_name,
    email,
    title,
    story,
    total,
    show,
    tell,
    reflection,
    helpfulness,  # NEW: overall agreement with explanations/highlights
    comments,
    # list of (idx, text, label, agree_int, highlight_str, explanation_str)
    sentence_rows,
):
    """
    NOTE: requires a TEXT column `helpfulness` in student_inputs.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()

        # parent row
        cur.execute(
            """
            INSERT INTO student_inputs
              (student_id, week_id,
               student_name, email, title, story,
               total_sentences, show_sentences, tell_sentences,
               reflection, helpfulness, comments)
            VALUES (%s, %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s)
            """,
            (
                student_id,
                week_id,
                student_name,
                email,
                title,
                story,
                total,
                show,
                tell,
                reflection,
                helpfulness,
                comments,
            ),
        )
        input_id = cur.lastrowid

        # sentence rows (with highlight_words + explanation)
        cur.executemany(
            """
            INSERT INTO student_sentences
              (input_id, week_id, sentence_idx, sentence_text, model_label,
               student_agree, highlight_words, explanation)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (input_id, week_id, idx, text, label, agree, highlights, expl)
                for (idx, text, label, agree, highlights, expl) in sentence_rows
            ],
        )

        conn.commit()
        return input_id
    except mysql.connector.Error as err:
        conn.rollback()
        st.error(f"⚠️ MySQL Error (rolled back): {err}")
        return None
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

# =========================
# Email
# =========================
def _masked(s: str) -> str:
    if not s:
        return ""
    if "@" in s:
        name, dom = s.split("@", 1)
        return (name[:2] + "…" if len(name) > 2 else "…") + "@" + dom
    return s[:2] + "…"


def send_feedback_email(email, student_name, title, summary, feedback_list, reflection, comment) -> bool:
    changed = sum(1 for item in feedback_list if not item["agree"])
    details = "🧾 Sentence-by-sentence feedback:\n"
    for item in feedback_list:
        status = "✅ Agreed" if item["agree"] else "❌ Did NOT agree"
        details += f"- [{item['label']}] {item['sentence']}\n  ➤ {status}\n\n"

    msg = EmailMessage()
    msg["Subject"] = f"📊 Feedback for Your Data Story: {title}"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = email
    msg["Reply-To"] = EMAIL_ADDRESS
    msg.set_content(
        f"""
Dear {student_name},

Thank you for submitting your data story titled "{title}". Our system analyzed your submission and identified a total of {summary["total_sentences"]} sentences. Of these, {summary["show_sentences"]} were categorized as 'Show' and {summary["tell_sentences"]} as 'Tell'. You disagreed with the model's classification on {changed} sentence(s).

{details}
Your comment:
"{comment if comment else 'No additional comment provided.'}"

Your reflection:
"{reflection if reflection else 'No reflection provided.'}"

Best regards,
The Data Story Feedback Team
"""
    )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(EMAIL_ADDRESS.strip(), EMAIL_PASSWORD.strip())
            smtp.send_message(msg)
        st.success(f"✅ Email sent to {_masked(email)}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        st.error(
            "❌ Gmail auth failed. Use a 16-char App Password (no spaces), "
            "approve the security alert, or regenerate the app password."
        )
        st.caption(f"Login user: {_masked(EMAIL_ADDRESS)} | {e.smtp_error!r}")
        return False
    except Exception as e:
        st.error("❌ Failed to send email.")
        st.exception(e)
        return False

# =========================
# Admin + Week
# =========================
def current_week_default() -> int:
    try:
        start_str = st.secrets.get("COURSE_START_DATE", None)
        if start_str:
            from datetime import date
            start = date.fromisoformat(str(start_str))
            week = (date.today() - start).days // 7 + 1
            return max(1, week)
        return int(st.secrets.get("CURRENT_WEEK", 5))
    except Exception:
        return 5


def current_week_image() -> str:
    try:
        images = list(st.secrets.get("WEEK_IMAGES", []))
        week = st.session_state.week_number
        if images and 1 <= week <= len(images):
            return f"images/{images[week - 1]}"
    except Exception:
        pass
    return "images/dog_walk.png"


if "week_number" not in st.session_state:
    st.session_state.week_number = current_week_default()

st.markdown(f"**Week:** {int(st.session_state.week_number)}")

# =========================
# UI flow
# =========================
st.title("✨ Show or Tell Prediction App ✨")
st.markdown("### Data Story Prompt")
st.image(current_week_image(), caption="Use this chart to write your data story.")
st.write("---")

if "page" not in st.session_state:
    st.session_state.page = "input"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# =========================
# INPUT PAGE
# =========================
if st.session_state.page == "input":
    student_name = st.text_input("Enter your name:")
    email = st.text_input("Enter your email:")
    title = st.text_input("Enter a title for your data story:")
    input_text = st.text_area("Write your data story here:")
    stories = [input_text.strip()] if input_text.strip() else []

    if st.button("Analyze"):
        if not student_name or not email or not title:
            st.error("Please fill in your name, email, and story title before continuing.")
        elif stories:
            _sid = get_or_create_student(student_name, email)
            _wid = get_or_create_week(st.session_state.week_number)
            if _sid and _wid and has_existing_submission(_sid, _wid):
                st.error(
                    f"You've already submitted for Week {int(st.session_state.week_number)}. "
                    "Resubmissions are closed."
                )
            else:
                st.session_state.page = "results"
                st.session_state.stories = stories
                st.session_state.student_name = student_name
                st.session_state.student_email = email
                st.session_state.story_title = title
                for key in [
                    "analysis_sentences",
                    "analysis_predictions",
                    "student_feedback",
                    "sentence_rows",
                    "total_sentences",
                    "show_sentences",
                    "tell_sentences",
                    "common_reason",
                    "helpfulness",
                    "analysis_results",
                ]:
                    st.session_state.pop(key, None)

# =========================
# RESULTS + REFLECTION
# =========================
if st.session_state.page == "results":
    stories = st.session_state.stories
    name = st.session_state.student_name
    email_addr = st.session_state.student_email
    story_title = st.session_state.story_title
    week_number = int(st.session_state.week_number)

    model, vectorizer = load_model_and_vectorizer()

    if not st.session_state.analysis_done:
        st.markdown("## Sentence Analysis")
        st.markdown("#### Note: the underlined words are used to explain why your sentence is show or tell.")
        # 🔄 Show spinner while heavy analysis runs
        with st.spinner("Analyzing your story and generating explanations..."):
            # compute sentences + predictions once
            if "analysis_sentences" not in st.session_state:
                all_sentences = []
                all_predictions = []

                for story in stories:
                    sents = smart_sentence_tokenize(story)
                    preds = predict_sentences(sents, model, vectorizer)
                    all_sentences.extend(sents)
                    all_predictions.extend(preds)

                st.session_state.analysis_sentences = all_sentences
                st.session_state.analysis_predictions = all_predictions

            # local copies
            sentences = st.session_state.analysis_sentences
            predictions = st.session_state.analysis_predictions

            # --- compute highlights + explanations ONCE ---
            if "analysis_results" not in st.session_state:
                embedding_model = load_embedding_model()
                analysis_results = []

                for sentence, label in zip(sentences, predictions):
                    # ✅ use raw model label (0 = Show, 1 = Tell)
                    stage_id = int(label)

                    hl = ufunc.get_highlights_with_embeddings(
                        sentence,
                        stage_id,
                        embedding_model,
                        threshold=0.47,
                    )

                    explanation = None
                    if OPENROUTER_API_KEY:
                        prompt = f"""
You are an expert in data storytelling.

Definitions:
- "Show" statements are DESCRIPTIVE – they describe what is visible in the data/chart.
- "Tell" statements are INTERPRETIVE – they make claims, interpretations, or conclusions beyond what's directly visible.

Explain in 1–2 sentences why this sentence is classified as "{hl['stage']}"
FOCUS ON THE CHARACTERISTICS OF THE SENTENCE THAT MATCH THIS CLASSIFICATION TYPE.
PLEASE DO NOT argue against the classification; instead, justify why it fits this category.
Sentence: "{sentence}"
"""
                        explanation = ufunc.call_openrouter_llm(prompt, OPENROUTER_API_KEY)

                    analysis_results.append({
                        "sentence": sentence,
                        "type": hl["stage"],
                        "highlights": hl["highlights"],
                        "explanation": explanation,
                    })

                st.session_state.analysis_results = analysis_results

        # after spinner, reuse cached data
        sentences = st.session_state.analysis_sentences
        predictions = st.session_state.analysis_predictions
        analysis_results = st.session_state.analysis_results

        feedback_data, sentence_rows = [], []
        total = len(sentences)
        show = sum(1 for p in predictions if p == 0)
        tell = sum(1 for p in predictions if p == 1)

        for i, (sent, label) in enumerate(zip(sentences, predictions)):
            label_text = "Show" if label == 0 else "Tell"
            color = "green" if label == 0 else "red"

            sentence_result = analysis_results[i]
            highlight_words = sentence_result["highlights"]
            explanation = sentence_result["explanation"]

            highlighted_html = highlight_sentence(sent, highlight_words)

            st.markdown(
                f"<span style='color:{color}'><b><u>{label_text}:</u></b> {highlighted_html}</span>",
                unsafe_allow_html=True,
            )

            if explanation:
                st.info(f"Explanation: {explanation}")
            elif not OPENROUTER_API_KEY:
                st.caption("Explanation unavailable (OPENROUTER_API_KEY not set).")

            agree = st.checkbox(
                "I agree with the model's label", key=f"agree_{i}"
            )

            feedback_data.append(
                {"sentence": sent, "label": label_text, "agree": agree}
            )

            highlight_str = ", ".join(highlight_words) if highlight_words else ""
            explanation_str = explanation or ""

            sentence_rows.append(
                (i, sent, label_text, 1 if agree else 0, highlight_str, explanation_str)
            )

        # Persist for DB + email
        st.session_state.student_feedback = feedback_data
        st.session_state.sentence_rows = sentence_rows
        st.session_state.total_sentences = total
        st.session_state.show_sentences = show
        st.session_state.tell_sentences = tell

        st.markdown("## Summary")
        st.write(f"Week: {week_number}")
        st.write(f"Total Sentences: {total}")
        st.write(f"Show Sentences: {show}")
        st.write(f"Tell Sentences: {tell}")

        fig, ax = plt.subplots()
        ax.bar(["Show", "Tell"], [show, tell])
        ax.set_ylabel("Number of Sentences")
        ax.set_title("Show vs Tell Breakdown")
        st.pyplot(fig)

        # NEW: overall agreement with explanations & highlights
        st.markdown("## Agreement with Explanations & Highlights")
        st.session_state.helpfulness = st.text_area(
            "Do you agree with the explanations and the highlighted key words (if applicable)? "
            "Please explain why or why not.",
            value=st.session_state.get("helpfulness", ""),
        )

        st.markdown("## Comment")
        st.session_state.common_reason = st.text_area(
            "Add any other comments you would like to add",
            value=st.session_state.get("common_reason", ""),
        )

        if st.button("Next: Reflection & Email"):
            st.session_state.analysis_done = True
            st.session_state.feedback_complete = True

    elif st.session_state.get("feedback_complete"):
        st.markdown("### Reflection")
        reflection = st.text_area(
            "What did you learn from this feedback?", key="reflection"
        )

        if st.button("Submit Feedback & Send Email"):

            student_id = get_or_create_student(name, email_addr)
            week_id = get_or_create_week(week_number)
            if not student_id or not week_id:
                st.error("Could not resolve student/week. Aborting save.")
            else:
                if has_existing_submission(student_id, week_id):
                    st.error(
                        f"You've already submitted for Week {week_number}. "
                        "Resubmissions are closed."
                    )
                    st.stop()

                input_id = insert_submission_and_sentences(
                    student_id,
                    week_id,
                    name,
                    email_addr,
                    story_title,
                    st.session_state.stories[0],
                    st.session_state.total_sentences,
                    st.session_state.show_sentences,
                    st.session_state.tell_sentences,
                    reflection,
                    st.session_state.helpfulness,   # <-- new field
                    st.session_state.common_reason,
                    st.session_state.sentence_rows,
                )

                if input_id:
                    summary = {
                        "total_sentences": st.session_state.total_sentences,
                        "show_sentences": st.session_state.show_sentences,
                        "tell_sentences": st.session_state.tell_sentences,
                    }
                    email_ok = send_feedback_email(
                        email_addr,
                        name,
                        story_title,
                        summary,
                        st.session_state.student_feedback,
                        reflection,
                        st.session_state.common_reason,
                    )

                    if email_ok:
                        st.success("✅ Feedback saved and email sent!")
                    else:
                        st.warning(
                            "Feedback saved to the database, but the email could not be sent."
                        )
                else:
                    st.error("❌ Could not save submission. Email not sent.")

        if st.button("Restart"):
            st.session_state.clear()
            st.rerun()



