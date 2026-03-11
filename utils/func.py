from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
import json

def test ():
    return 2

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


SHOW_INDICATORS = [
    # Visual chart types
    "bar chart", "line graph", "pie chart", "scatter plot", "histogram",
    "visualization", "figure", "image", "graphic", "infographic",

    # Visual components - axes and labels
    "title", "axis", "x-axis", "y-axis", "horizontal axis", "vertical axis",
    "label", "legend", "scale", "grid", "tick mark", "annotation",

    # Visual encoding elements
    "color", "size", "shape", "position", "height", "width", "length",
    "symbol", "marker", "icon", "bar", "line", "point", "area",

    # Descriptive verbs (literal reading)
    "displays", "shows", "depicts", "illustrates", "represents", "presents",
    "contains", "includes", "features", "exhibits", "portrays",

    # Identification/Recognition words
    "labeled", "labeled as", "titled", "named", "called", "identified",
    "positioned", "located", "placed", "arranged", "organized",

    # Component identification
    "visual", "chart", "graph", "plot", "diagram", "table",
    "component", "element", "part", "section", "category",

    # Data reading (literal)
    "data point", "value", "number", "measurement", "reading",
    "percentage", "percent", "count", "total", "sum",

    # Visual properties
    "horizontal", "vertical", "ascending", "descending",
    "visible", "shown", "displayed", "indicated",

    # Sense-making (basic)
    "can be seen", "is visible", "is shown", "appears",
    "is depicted", "is illustrated", "is represented"
]

TELL_INDICATORS = [
    # Comparison words (reading between)
    "higher than", "lower than", "compared to", "versus", "vs",
    "exceeds", "surpasses", "falls below", "greater than", "less than",
    "more than", "fewer than", "equal to", "similar to", "different from",
    "in contrast", "whereas", "while", "although", "however",

   # Pattern identification (reading between)
    "trend", "pattern", "increase", "decrease", "rise", "fall",
    "growth", "decline", "fluctuation", "variation", "change",
    "consistent", "inconsistent", "stable", "unstable", "steady",
    "correlation", "relationship", "association", "connection",

    # Extremes and outliers (reading between)
    "highest", "lowest", "maximum", "minimum", "peak", "trough",
    "outlier", "anomaly", "exception", "unusual", "extreme",
    "most", "least", "top", "bottom", "best", "worst",

    # Ranges and distributions (reading between)
    "range", "spread", "distribution", "cluster", "concentration",
    "scattered", "grouped", "concentrated", "dispersed",

    # Approximations (reading between)
    "approximately", "around", "roughly", "about", "nearly",
    "close to", "estimate", "estimation",

    # Aggregation terms
    "majority", "minority", "proportion", "percentage", "fraction",
    "most", "some", "few", "several", "many",
    "dominant", "dominate", "prevalent", "common", "rare",

    # Inference words (reading beyond)
    "suggests", "implies", "indicates", "reveals", "demonstrates",
    "shows that", "means", "signifies", "reflects", "points to",
    "likely", "probably", "possibly", "appears to", "seems to",
    "may", "might", "could", "would", "should",

    # Causal/explanatory (reading beyond)
    "because", "due to", "as a result", "therefore", "thus",
    "consequently", "hence", "leads to", "causes", "results in",
    "explains", "accounts for", "attributed to", "from",
    "impact", "effect", "influence", "affect", "consequence",

    # Significance/importance (reading beyond)
    "significant", "important", "notable", "remarkable", "noteworthy",
    "substantial", "considerable", "major", "minor", "key",
    "critical", "crucial", "essential", "meaningful",

    # Interpretation beyond literal
    "story", "narrative", "tells us", "reveals that",
    "conclusion", "finding", "insight", "observation",
    "interpretation", "meaning", "implication",

    # Knowledge-based terms (reading beyond)
    "expected", "unexpected", "surprising", "typical", "atypical",
    "normal", "abnormal", "usual", "unusual", "standard",
    "known", "unknown", "familiar", "unfamiliar",

    # Relative positioning
    "take up", "occupy", "make up", "comprise", "constitute",
    "represent", "account for",

    # Probability/likelihood
    "more likely", "less likely", "unlikely", "chance", "probability",
    "tend to", "inclined to", "prone to",

    # Popularity/frequency
    "popular", "common", "frequent", "rare", "infrequent",
    "often", "seldom", "sometimes", "always", "never",

    # Social/contextual analysis (reading beyond)
    "discrimination", "bias", "disparity", "inequality", "gap",
    "diversity", "representation", "underrepresented", "overrepresented",
    "access", "opportunity", "barrier", "challenge",

    # Temporal patterns
    "over time", "throughout", "during", "since", "until",
    "continues", "persists", "remains", "maintained",

    # Degree modifiers
    "very", "extremely", "highly", "significantly", "slightly",
    "somewhat", "moderately", "considerably", "substantially"
]
TYPE_MAP = {0: "Show", 1: "Tell"}
# RACE-RELATED TERMS TO EXCLUDE
RACE_TERMS = {
    'white', 'black', 'asian', 'hispanic', 'latino', 'latina', 'latinx',
    'african', 'caucasian', 'race', 'races', 'racial', 'ethnicity',
    'ethnic', 'people of color', 'minority group', 'minorities',
}


def contains_race_terms(phrase):
    """Check if phrase contains any race-related terms."""
    phrase_lower = phrase.lower()

    # Check for exact race terms
    for term in RACE_TERMS:
        if term in phrase_lower:
            return True

    # Check for "people" when it appears with demographic context
    # Allow "people" in phrases like "popular for people" but not "white people"
    words = phrase_lower.split()
    if 'people' in words:
        idx = words.index('people')
        # Check if race term appears near "people"
        context_window = 2
        start = max(0, idx - context_window)
        end = min(len(words), idx + context_window + 1)
        context = ' '.join(words[start:end])

        for term in RACE_TERMS:
            if term in context:
                return True

    return False


def ultra_aggressive_deduplicate(highlights):
    """
    NO content word can appear in multiple highlights.
    Also filters out race-related terms.
    """
    if not highlights:
        return []

    # First, filter out race terms
    filtered = [h for h in highlights if not contains_race_terms(h)]

    if not filtered:
        return []

    unique = list(dict.fromkeys([h.lower().strip() for h in filtered]))
    sorted_hl = sorted(unique, key=len, reverse=True)

    stopwords = {'the', 'a', 'an', 'of', 'in', 'to', 'for', 'and', 'or', 'is', 'are',
                 'at', 'by', 'on', 'with', 'from', 'as', 'all', 'have', 'has', 'who'}

    kept = []
    used_words = set()

    for current in sorted_hl:
        current_words = set(w for w in current.split()
                           if w not in stopwords and len(w) > 1)

        if not current_words:
            continue

        # Skip if ANY content word is already used
        if current_words & used_words:
            continue

        kept.append(current)
        used_words.update(current_words)

        if len(kept) >= 6:
            break

    return kept


def extract_ngrams(sentence, n=3):
    words = sentence.lower().split()
    ngrams = []

    # 3-word phrases
    for i in range(len(words) - 2):
        ngrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")

    # 2-word phrases
    for i in range(len(words) - 1):
        ngrams.append(f"{words[i]} {words[i+1]}")

    # Single words (filter stopwords)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'in', 'of', 'to', 'for', 'and', 'or'}
    for word in words:
        if word not in stopwords and len(word) > 2:
            ngrams.append(word)

    return ngrams


def get_highlights_with_embeddings(sentence: str, stage_type: int,
                                   model, threshold: float = 0.47,
                                   max_highlights: int = 3):
                                   
    type_name = TYPE_MAP.get(stage_type, "Unknown")
    
    if type_name == "Unknown" or not sentence.strip():
        return {'stage': type_name, 'highlights': []}

    indicators = SHOW_INDICATORS if stage_type == 1 else TELL_INDICATORS
    candidates = list(dict.fromkeys(extract_ngrams(sentence, n=3)))
    
    # Get embeddings
    candidate_embeddings = model.encode(candidates)
    indicator_embeddings = model.encode(indicators)
    similarities = cosine_similarity(candidate_embeddings, indicator_embeddings)
    max_similarities = similarities.max(axis=1)

    # Filter by threshold
    highlight_scores = []
    for i, candidate in enumerate(candidates):
        if max_similarities[i] >= threshold and candidate.lower() in sentence.lower():
            highlight_scores.append((candidate, max_similarities[i]))

    # Sort by score, then length
    highlight_scores.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    raw_highlights = [h[0] for h in highlight_scores]

    # Ultra-aggressive deduplication WITH race filter
    highlights = ultra_aggressive_deduplicate(raw_highlights)[:max_highlights]

    return {
        'stage': type_name,
        'highlights': highlights
    }

def call_openrouter_llm(prompt: str, api_key: str, model: str = "openai/gpt-4o-mini") -> str:
    """
    Call OpenRouter API with a prompt.
    
    Args:
        prompt: Your prompt text
        api_key: OpenRouter API key
        model: Model to use (default: "openai/gpt-4o-mini")
    
    Returns:
        The LLM's response as a string
    
    Example:
        response = call_openrouter_llm("Explain AI", "your-api-key")
        print(response)
    """
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )
    
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
