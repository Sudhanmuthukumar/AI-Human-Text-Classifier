import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from flask import Flask, render_template, request, jsonify
import joblib, re, nltk, numpy as np, textstat
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
from nrclex import NRCLex

app = Flask(__name__)
tfidf_vector = joblib.load("models/tfidf.pkl")
svd = joblib.load("models/svd.pkl")
log_reg = joblib.load("models/log_reg.pkl")

booster = xgb.Booster()
booster.load_model("models/xgb_booster.json")

REPLACE_BAD_WORD = re.compile(r'[/(){}\[\]\|@,;]')

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = REPLACE_BAD_WORD.sub(" ", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)


def extract_linguistic_features(text):
    readability = textstat.flesch_reading_ease(text)
    words = text.split()
    vocab_div = len(set(words)) / len(words) if words else 0
    avg_sent_len = np.mean([len(s.split()) for s in text.split('.') if s.strip()]) if '.' in text else len(words)
    return np.array([readability, vocab_div, avg_sent_len])



NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

def analyze_emotions(text):
    """Use NRCLex to extract emotion percentages and trigger words for the 8 NRC emotions."""
    emotion_obj = NRCLex(text)
    emotion_obj.load_raw_text(text)
    raw_scores = emotion_obj.raw_emotion_scores

    # Filter to only the 8 target emotions
    filtered = {e: raw_scores.get(e, 0) for e in NRC_EMOTIONS}
    total = sum(filtered.values())

    # Percentage distribution
    if total > 0:
        emotions_pct = {e: round((v / total) * 100, 1) for e, v in filtered.items()}
    else:
        emotions_pct = {e: 0 for e in NRC_EMOTIONS}

    # Dominant emotion
    if total > 0:
        dominant = max(filtered, key=filtered.get)
    else:
        dominant = "No emotion detected"

    # Words that triggered each emotion
    affect_dict = emotion_obj.affect_dict  # word -> list of emotions
    emotion_words = {e: [] for e in NRC_EMOTIONS}
    for word, emo_list in affect_dict.items():
        for emo in emo_list:
            if emo in emotion_words and word not in emotion_words[emo]:
                emotion_words[emo].append(word)

    # Remove empty entries
    emotion_words_filtered = {e: wds for e, wds in emotion_words.items() if wds}

    return {
        "dominant_emotion": dominant,
        "emotions": emotions_pct,
        "emotion_words": emotion_words_filtered
    }

def predict_text(text):
    cleaned = clean_text(text)
    tfidf_vec = tfidf_vector.transform([cleaned])
    svd_features = svd.transform(tfidf_vec)
    ling = extract_linguistic_features(text).reshape(1, -1)
    final_features = np.hstack([svd_features, ling])

    dmatrix = xgb.DMatrix(final_features)
    xgb_proba = booster.predict(dmatrix)
    lr_proba = log_reg.predict_proba(final_features)[:, 1]
    avg_proba = (xgb_proba + lr_proba) / 2

    ai_score = float(avg_proba[0])
    human_score = 1.0 - ai_score

    if 0.4 <= ai_score <= 0.6:
        label = "Uncertain"
    elif ai_score > 0.6:
        label = "AI"
    else:
        label = "Human"

    readability, vocab_div, avg_sent_len = ling[0]

    # Emotion analysis (independent of AI detection)
    emotion_data = analyze_emotions(text)

    result = {
        "label": label,
        "ai_score": round(ai_score * 100, 2),
        "human_score": round(human_score * 100, 2),
        "readability": round(float(readability), 2),
        "vocab_diversity": round(float(vocab_div), 2),
        "avg_sentence_length": round(float(avg_sent_len), 2)
    }
    result.update(emotion_data)
    return result

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    return jsonify(predict_text(text))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
