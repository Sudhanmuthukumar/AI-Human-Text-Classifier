document.addEventListener("DOMContentLoaded", () => {
  const checkBtn = document.getElementById("checkBtn");
  const clearBtn = document.getElementById("clearBtn");
  const userText = document.getElementById("userText");
  const resultCard = document.getElementById("result");
  const resultPlaceholder = document.getElementById("resultPlaceholder");
  const warningCard = document.getElementById("warningCard");
  const warningTitle = document.getElementById("warningTitle");
  const warningDesc = document.getElementById("warningDesc");
  const scoreText = document.getElementById("scoreText");
  const meterFill = document.getElementById("meterFill");
  const meterBox = document.getElementById("meterBox");
  const breakdown = document.getElementById("breakdown");
  const resultBadge = document.getElementById("resultBadge");

  // Helpers to toggle the three output states
  function showPlaceholder() {
    if (resultPlaceholder) resultPlaceholder.style.display = "flex";
    if (resultCard) resultCard.style.display = "none";
    if (warningCard) warningCard.style.display = "none";
  }

  function showWarning(title, desc) {
    if (resultPlaceholder) resultPlaceholder.style.display = "none";
    if (resultCard) resultCard.style.display = "none";
    if (warningCard) warningCard.style.display = "flex";
    if (warningTitle) warningTitle.textContent = title;
    if (warningDesc) warningDesc.textContent = desc;
  }

  function showResultCard() {
    if (resultPlaceholder) resultPlaceholder.style.display = "none";
    if (warningCard) warningCard.style.display = "none";
    if (resultCard) resultCard.style.display = "flex";
  }

  // Initial state
  showPlaceholder();

  // Restore saved text
  const saved = localStorage.getItem("userText");
  if (saved) userText.value = saved;


  // Render result
  function showResult(r) {
    showResultCard();

    const label = r.label.toLowerCase();
    resultBadge.textContent = r.label;
    resultBadge.className = `result-badge ${label}`;

    meterBox.className = `meter ${label}`;

    const confidence = Math.max(r.ai_score, r.human_score);
    scoreText.textContent = `${confidence}% confidence`;

    meterFill.style.width = r.ai_score + "%";

    breakdown.innerHTML = `
      <li class="metric-card">
        <div class="metric-label">Readability</div>
        <div class="metric-value">${r.readability}</div>
      </li>
      <li class="metric-card">
        <div class="metric-label">Vocabulary</div>
        <div class="metric-value">${r.vocab_diversity}</div>
      </li>
      <li class="metric-card">
        <div class="metric-label">Avg. Sentence</div>
        <div class="metric-value">${r.avg_sentence_length} <span style="font-size:12px;color:var(--text-tertiary)">words</span></div>
      </li>
    `;

    // === Emotion Analysis ===
    const emotionSection = document.getElementById("emotionSection");
    const dominantEmotion = document.getElementById("dominantEmotion");
    const emotionBars = document.getElementById("emotionBars");
    const emotionWords = document.getElementById("emotionWords");
    const emotionWordsSection = document.getElementById("emotionWordsSection");

    if (r.emotions && r.dominant_emotion) {
      emotionSection.style.display = "block";

      // Emoji map for emotions
      const emojiMap = {
        anger: "😠", anticipation: "🤔", disgust: "🤢", fear: "😨",
        joy: "😊", sadness: "😢", surprise: "😲", trust: "🤝"
      };

      dominantEmotion.textContent = `${emojiMap[r.dominant_emotion] || ""} ${r.dominant_emotion.charAt(0).toUpperCase() + r.dominant_emotion.slice(1)}`;

      // Emotion distribution bars
      let barsHtml = "";
      const emotions = r.emotions;
      for (const [emo, pct] of Object.entries(emotions)) {
        barsHtml += `
          <div class="emotion-bar-row">
            <span class="emotion-bar-label">${emojiMap[emo] || ""} ${emo.charAt(0).toUpperCase() + emo.slice(1)}</span>
            <div class="emotion-bar-track">
              <div class="emotion-bar-fill" style="width:${pct}%"></div>
            </div>
            <span class="emotion-bar-pct">${pct}%</span>
          </div>`;
      }
      emotionBars.innerHTML = barsHtml;

      // Emotion words
      if (r.emotion_words && Object.keys(r.emotion_words).length > 0) {
        emotionWordsSection.style.display = "block";
        let wordsHtml = "";
        for (const [emo, words] of Object.entries(r.emotion_words)) {
          wordsHtml += `
            <div class="emotion-word-group">
              <span class="emotion-word-label">${emojiMap[emo] || ""} ${emo.charAt(0).toUpperCase() + emo.slice(1)}</span>
              <span class="emotion-word-list">${words.join(", ")}</span>
            </div>`;
        }
        emotionWords.innerHTML = wordsHtml;
      } else {
        emotionWordsSection.style.display = "none";
      }
    } else {
      emotionSection.style.display = "none";
    }
  }

  // Analyze
  checkBtn.addEventListener("click", async () => {
    const text = userText.value.trim();

    if (!text) {
      showWarning("No text entered", "Paste or type some text on the left, then click Analyze.");
      return;
    }

    if (text.length < 250) {
      const remaining = 250 - text.length;
      showWarning(
        "Not enough text",
        `You need at least 250 characters for accurate analysis. Add ${remaining} more character${remaining === 1 ? "" : "s"}.`
      );
      return;
    }

    localStorage.setItem("userText", text);

    // Loading state
    showResultCard();
    resultBadge.textContent = "Analyzing";
    resultBadge.className = "result-badge";
    scoreText.textContent = "Reading your text…";
    meterFill.style.width = "0%";
    breakdown.innerHTML = "";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await res.json();
      if (data.error) {
        showWarning("Analysis failed", data.error);
        return;
      }

      localStorage.setItem("lastResult", JSON.stringify(data));
      showResult(data);
    } catch (err) {
      console.error(err);
      showWarning("Connection error", "Could not reach the server. Please make sure it's running and try again.");
    }
  });

  // Clear
  clearBtn.addEventListener("click", () => {
    userText.value = "";
    showPlaceholder();
    localStorage.removeItem("userText");
    localStorage.removeItem("lastResult");
  });
});
