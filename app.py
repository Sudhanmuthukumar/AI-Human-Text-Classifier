from flask import Flask, render_template, request, jsonify
import joblib, re, nltk, numpy as np, textstat
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xgboost as xgb

app = Flask(__name__)

tfidf_vector = joblib.load("tfidf.pkl")
svd = joblib.load("svd.pkl")
log_reg = joblib.load("log_reg.pkl")
booster = xgb.Booster()
booster.load_model("xgb_booster.json")

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

sentiment_lexicon = {

    "joy": (1, 3), "bliss": (1, 3), "ecstatic": (1, 3), "jubilant": (1, 3),
    "euphoric": (1, 3), "triumph": (1, 3), "love": (1, 3), "adore": (1, 3),
    "cherish": (1, 3), "perfect": (1, 3),

    "wonderful": (1, 2), "amazing": (1, 2), "awesome": (1, 2),
    "fantastic": (1, 2), "brilliant": (1, 2),
    "laugh": (1, 2), "laughing": (1, 2),
    "happy": (1, 2), "delight": (1, 2),
    "elated": (1, 2), "glad": (1, 2),
    "excited": (1, 2), "enthusiastic": (1, 2),
    "proud": (1, 2), "hope": (1, 2),
    "satisfaction": (1, 2),

    "cheerful": (1, 1), "content": (1, 1), "optimistic": (1, 1),
    "energetic": (1, 1), "peaceful": (1, 1),
    "confident": (1, 1), "successful": (1, 1),
    "thankful": (1, 1), "loving": (1, 1),
    "inspired": (1, 1), "like": (1, 1),
    "enjoy": (1, 1), "appreciate": (1, 1),
    "smile": (1, 1), "cool": (1, 1),

    "sadness": (-1, 3), "agony": (-1, 3), "grief": (-1, 3),
    "despair": (-1, 3), "misery": (-1, 3),
    "terrible": (-1, 3), "horrible": (-1, 3),
    "worst": (-1, 3), "devastated": (-1, 3),

    "sad": (-1, 2), "depressed": (-1, 2),
    "lonely": (-1, 2), "gloomy": (-1, 2),
    "hopeless": (-1, 2), "heartbroken": (-1, 2),
    "fail": (-1, 2), "failure": (-1, 2),
    "cry": (-1, 2),

    "unhappy": (-1, 1), "disappointed": (-1, 1),
    "hurt": (-1, 1), "ashamed": (-1, 1),
    "guilty": (-1, 1), "empty": (-1, 1),
    "bored": (-1, 1), "tired": (-1, 1),
    "bad": (-1, 1),

    "anger": (-1, 3), "furious": (-1, 3),
    "rage": (-1, 3), "hate": (-1, 3),
    "mad": (-1, 2), "frustrated": (-1, 2),
    "annoyed": (-1, 2), "hostile": (-1, 2),
    "upset": (-1, 1),

    "fear": (-1, 3), "panic": (-1, 3),
    "terrified": (-1, 3),
    "scared": (-1, 2), "afraid": (-1, 2),
    "anxious": (-1, 1), "worried": (-1, 1),
    "stressed": (-1, 1),

    "disgust": (-1, 3), "disgusted": (-1, 2),
    "gross": (-1, 2), "nasty": (-1, 2),
    "awful": (-1, 1)
}

intensifiers = {
    "very": 1.5, "really": 1.5, "extremely": 2.0, "absolutely": 2.0,
    "more": 1.2, "most": 1.5, "so": 1.2,
    "slightly": 0.5, "little": 0.5, "hardly": 0.5
}

negations = {
    "not", "no", "never", "cannot", "can't",
    "don't", "didn't", "doesn't",
    "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
    "without", "hardly"
}

stoppers = {
    "but", "however", "though", "although", "yet"
}

def sentiment_from_text(text):
    clauses = re.split(r"[,.]", text.lower())
    clause_scores = []

    for clause in clauses:
        words = re.findall(r"\b\w+'\w+|\w+\b", clause)

        score = 0.0
        neg_count = 0
        intensity = 1.0

        for word in words:
            if word in stoppers:
                score *= 0.5
                neg_count = 0
                intensity = 1.0
                continue

            if word in negations:
                neg_count += 1
                continue

            if word in intensifiers:
                intensity *= intensifiers[word]
                continue

            if word in sentiment_lexicon:
                polarity, weight = sentiment_lexicon[word]
                value = polarity * weight * intensity

                if neg_count % 2 == 1:
                    value *= -1
                    value *= 0.8

                score += value
                neg_count = 0
                intensity = 1.0

        clause_scores.append(score)

    final_score = 0.0
    for i, sc in enumerate(clause_scores):
        final_score += sc * (1 + i * 0.5)

    if final_score > 0.5:
        return "Positive"
    elif final_score < -0.5:
        return "Negative"
    else:
        return "Neutral"

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

    sentiment = sentiment_from_text(text)
    readability, vocab_div, avg_sent_len = ling[0]

    return {
        "label": label,
        "ai_score": round(ai_score * 100, 2),
        "human_score": round(human_score * 100, 2),
        "readability": round(float(readability), 2),
        "vocab_diversity": round(float(vocab_div), 2),
        "avg_sentence_length": round(float(avg_sent_len), 2),
        "sentiment": sentiment
    }

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
