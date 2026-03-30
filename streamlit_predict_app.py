import os
import re
import json
import smtplib
import mysql.connector
import nltk
import streamlit as st
import matplotlib.pyplot as plt
from email.message import EmailMessage
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
    print(EMAIL_ADDRESS, EMAIL_PASSWORD)
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
# OpenRouter key (for classification + explanations)
# =========================
OPENROUTER_API_KEY = (get_secret("OPENROUTER_API_KEY") or "").strip()
if not OPENROUTER_API_KEY:
    st.error("Missing OPENROUTER_API_KEY in Streamlit secrets.")
    st.stop()

# =========================
# OpenRouter sentence classifier
# =========================
def classify_sentences_with_openrouter(sentences: list[str], api_key: str) -> list[int]:
    """
    Classify sentences as Show (0) or Tell (1) using OpenRouter AI.
    Returns a list of integers: 0 = Show, 1 = Tell.
    """
    if not sentences:
        return []

    numbered = "\n".join(f'{i+1}. "{s}"' for i, s in enumerate(sentences))
    prompt = (
        'You are classifying items from student data stories as "Show", "Tell", or "Not a sentence".\n\n'
        "Definitions:\n"
        '- "Show" sentences are DESCRIPTIVE – they describe what is literally visible in the data/chart.\n'
        '- "Tell" sentences are INTERPRETIVE – they make claims, draw conclusions, or interpret beyond what\'s directly visible.\n'
        '- "Not a sentence" items are fragments, titles, headings, labels, or any text that is not a grammatically complete sentence.\n\n'
        "Classify each item below. Return ONLY a JSON array of labels, one per item, "
        'where each label is exactly "Show", "Tell", or "Not a sentence".\n'
        'Example output for 4 items: ["Show", "Tell", "Not a sentence", "Show"]\n\n'
        f"Items:\n{numbered}"
    )

    try:
        raw = ufunc.call_openrouter_llm(prompt, api_key)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            labels = json.loads(match.group())
            if len(labels) == len(sentences):
                def _map(l):
                    s = str(l).strip().lower()
                    if s == "show": return 0
                    if s == "not a sentence": return 2
                    return 1
                return [_map(l) for l in labels]
    except Exception as e:
        print(f"[OpenRouter classify error] {e}")

    # fallback: classify as Tell
    return [1] * len(sentences)


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
    sentence_fragment,
    agree_fragment,
    disagree_fragment,
    reflection,
    helpfulness,
    comments,
    # list of (idx, text, label, agree_int, highlight_str, explanation_str)
    sentence_rows,
):
    # Always compute total from components to satisfy the DB check constraint
    total = show + tell + sentence_fragment

    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()

        # delete any existing submission for this student+week (allow resubmission)
        cur.execute(
            "SELECT input_id FROM student_inputs WHERE student_id=%s AND week_id=%s",
            (student_id, week_id),
        )
        existing = cur.fetchone()
        if existing:
            cur.execute("DELETE FROM student_sentences WHERE input_id=%s", (existing[0],))
            cur.execute("DELETE FROM student_inputs WHERE input_id=%s", (existing[0],))

        # parent row
        cur.execute(
            """
            INSERT INTO student_inputs
              (student_id, week_id,
               student_name, email, title, story,
               total_sentences, show_sentences, tell_sentences, sentence_fragment,
               agree_fragment, disagree_fragment,
               reflection, helpfulness, comments)
            VALUES (%s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
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
                sentence_fragment,
                agree_fragment,
                disagree_fragment,
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
            from datetime import date, datetime, timedelta
            from zoneinfo import ZoneInfo
            start = date.fromisoformat(str(start_str))
            # Weeks change at noon Maryland time — use Eastern Time explicitly,
            # then subtract 12h so the day flips at 12:00 PM ET (not server time)
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(tz=eastern)
            effective_date = (now_et - timedelta(hours=12)).date()
            week = (effective_date - start).days // 7 + 1
            return max(1, week)
        return int(st.secrets.get("CURRENT_WEEK", 1))
    except Exception:
        return 1


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
            # if _sid and _wid and has_existing_submission(_sid, _wid):
            #     st.error(
            #         f"You've already submitted for Week {int(st.session_state.week_number)}. "
            #         "Resubmissions are closed."
            #     )
            # else:
            if True:
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

    if not st.session_state.analysis_done:
        st.markdown("## Sentence Analysis")
        st.markdown("#### Note: the underlined words are used to explain why your sentence is show or tell.")
        # 🔄 Show spinner while heavy analysis runs
        with st.spinner("Analyzing your story and generating explanations..."):
            # compute sentences + predictions once
            if "analysis_sentences" not in st.session_state:
                all_sentences = []

                for story in stories:
                    sents = smart_sentence_tokenize(story)
                    all_sentences.extend(sents)

                all_predictions = classify_sentences_with_openrouter(
                    all_sentences, OPENROUTER_API_KEY
                )

                st.session_state.analysis_sentences = all_sentences
                st.session_state.analysis_predictions = all_predictions

            # local copies
            sentences = st.session_state.analysis_sentences
            predictions = st.session_state.analysis_predictions

            # --- compute highlights + explanations ONCE ---
            if "analysis_results" not in st.session_state:
                analysis_results = []

                for sentence, label in zip(sentences, predictions):
                    if label == 2:
                        # Not a sentence — no highlights, explain why
                        explanation = None
                        if OPENROUTER_API_KEY:
                            prompt = (
                                'Explain in 1–2 sentences why this text is NOT a complete sentence '
                                '(e.g. it is a fragment, title, heading, or label). '
                                f'Text: "{sentence}"'
                            )
                            explanation = ufunc.call_openrouter_llm(prompt, OPENROUTER_API_KEY)
                        analysis_results.append({
                            "sentence": sentence,
                            "type": "Not a sentence",
                            "highlights": [],
                            "explanation": explanation,
                        })
                        continue

                    # Show (0) or Tell (1) — ask LLM for highlights + explanation together
                    label_name = "Show" if label == 0 else "Tell"
                    highlights, explanation = [], None
                    if OPENROUTER_API_KEY:
                        prompt = (
                            'You are an expert in data storytelling.\n\n'
                            'Definitions:\n'
                            '- "Show" sentences are DESCRIPTIVE – they describe what is literally visible in the data/chart.\n'
                            '- "Tell" sentences are INTERPRETIVE – they make claims, draw conclusions, or interpret beyond what\'s directly visible.\n\n'
                            f'This sentence has been classified as "{label_name}".\n\n'
                            '1. Identify 1–3 short key words or phrases (max 3 words each) directly from the sentence that best indicate it is '
                            f'"{label_name}". Only use words that appear verbatim in the sentence.\n'
                            '2. Explain in 1–2 sentences why this sentence is classified as '
                            f'"{label_name}". Focus on why it fits this category.\n\n'
                            'Return ONLY a JSON object in this exact format (no extra text):\n'
                            '{"highlights": ["phrase1", "phrase2"], "explanation": "Your explanation here."}\n\n'
                            f'Sentence: "{sentence}"'
                        )
                        try:
                            raw = ufunc.call_openrouter_llm(prompt, OPENROUTER_API_KEY)
                            match = re.search(r"\{.*\}", raw, re.DOTALL)
                            if match:
                                parsed = json.loads(match.group())
                                highlights = parsed.get("highlights", [])
                                explanation = parsed.get("explanation", None)
                        except Exception as e:
                            print(f"[LLM highlight+explain error] {e}")

                    analysis_results.append({
                        "sentence": sentence,
                        "type": label_name,
                        "highlights": highlights,
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
        not_sentence = sum(1 for p in predictions if p == 2)

        for i, (sent, label) in enumerate(zip(sentences, predictions)):
            if label == 0:
                label_text, color = "Show", "green"
            elif label == 1:
                label_text, color = "Tell", "red"
            else:
                label_text, color = "Sentence Fragment", "gray"

            sentence_result = analysis_results[i]
            highlight_words = sentence_result["highlights"]
            explanation = sentence_result["explanation"]

            highlighted_html = highlight_sentence(sent, highlight_words)

            st.markdown(
                f"<span style='color:{color}'><b>{label_text}:</b> {highlighted_html}</span>",
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

        agree_fragment = sum(
            1 for (_, _, lbl, agr, _, _) in sentence_rows
            if lbl == "Sentence Fragment" and agr == 1
        )
        disagree_fragment = sum(
            1 for (_, _, lbl, agr, _, _) in sentence_rows
            if lbl == "Sentence Fragment" and agr == 0
        )

        # Persist for DB + email
        st.session_state.student_feedback = feedback_data
        st.session_state.sentence_rows = sentence_rows
        st.session_state.total_sentences = show + tell + not_sentence
        st.session_state.show_sentences = show
        st.session_state.tell_sentences = tell
        st.session_state.sentence_fragment = not_sentence
        st.session_state.agree_fragment = agree_fragment
        st.session_state.disagree_fragment = disagree_fragment

        st.markdown("## Summary")
        st.write(f"Week: {week_number}")
        st.write(f"Total Sentences: {total}")
        st.write(f"Show Sentences: {show}")
        st.write(f"Tell Sentences: {tell}")
        st.write(f"Sentence Fragment: {not_sentence}")

        fig, ax = plt.subplots()
        ax.bar(["Show", "Tell", "Sentence Fragment"], [show, tell, not_sentence])
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

        st.markdown("## Reflection")
        reflection = st.text_area(
            "What did you learn from this feedback?",
            value=st.session_state.get("reflection", ""),
            key="reflection_input"
        )

        if st.button("Submit Feedback & Send Email"):
            student_id = get_or_create_student(name, email_addr)
            week_id = get_or_create_week(week_number)
            if not student_id or not week_id:
                st.error("Could not resolve student/week. Aborting save.")
            # elif has_existing_submission(student_id, week_id):
            #     st.error(
            #         f"You've already submitted for Week {week_number}. "
            #         "Resubmissions are closed."
            #     )
            else:
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
                    st.session_state.sentence_fragment,
                    st.session_state.agree_fragment,
                    st.session_state.disagree_fragment,
                    reflection,
                    st.session_state.helpfulness,
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




