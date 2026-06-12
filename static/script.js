document.addEventListener("DOMContentLoaded", () => {
  const checkBtn = document.getElementById("checkBtn");
  const clearBtn = document.getElementById("clearBtn");
  const userText = document.getElementById("userText");
  const resultCard = document.getElementById("result");
  const scoreText = document.getElementById("scoreText");
  const meterFill = document.getElementById("meterFill");
  const breakdown = document.getElementById("breakdown");
  const warningMsg = document.getElementById("warningMsg");

  const exampleHumanBtn = document.getElementById("exampleHumanBtn");
  const exampleAiBtn = document.getElementById("exampleAiBtn");

  // Reset result card on load
  resultCard.style.display = "none";
  scoreText.textContent = "";
  breakdown.innerHTML = "";

  // Load saved text
  if (localStorage.getItem("userText")) {
    userText.value = localStorage.getItem("userText");
  }

  // Hide warning on input
  userText.addEventListener("input", () => {
    if (userText.value.trim().length > 0) {
      warningMsg.style.display = "none";
    }
  });

  // Sentiment display helper
  function getSentimentDisplay(sentiment) {
    switch (sentiment) {
      case "Positive": return { emoji: "😊", color: "green" };
      case "Negative": return { emoji: "😡", color: "red" };
      default: return { emoji: "😐", color: "gray" };
    }
  }

  // Display result
  function displayResult(result) {
    resultCard.style.display = "block";
    scoreText.textContent = `Prediction: ${result.label} | Confidence: ${Math.max(result.ai_score, result.human_score)}%`;
    meterFill.style.width = result.ai_score + "%";

    const { emoji, color } = getSentimentDisplay(result.sentiment);

    breakdown.innerHTML = `<br>
      <li><strong>Readability:</strong> ${result.readability}</li>
      <br><li><strong>Vocabulary Diversity:</strong> ${result.vocab_diversity}</li>
      <br><li><strong>Average Sentence Length:</strong> ${result.avg_sentence_length} words</li>
      <br><li><strong>Sentiment:</strong> <span style="color:${color};">${result.sentiment} ${emoji}</span></li>
    `;
  }

  // Check button
  checkBtn.addEventListener("click", async () => {
    const text = userText.value.trim();
    if (!text) {
      warningMsg.style.display = "block";
      resultCard.style.display = "none";
      breakdown.innerHTML = "";
      return;
    }

    if (text.length < 250) {
      alert("Please enter at least 250 characters.");
      return;
    }

    warningMsg.style.display = "none";
    localStorage.setItem("userText", text);

    scoreText.textContent = "Analyzing...";
    resultCard.style.display = "block";
    meterFill.style.width = "0%";
    breakdown.textContent = "";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const result = await response.json();
      if (result.error) {
        alert(result.error);
        resultCard.style.display = "none";
        return;
      }

      localStorage.setItem("lastResult", JSON.stringify(result));
      displayResult(result);
    } catch (err) {
      console.error(err);
      alert("Error connecting to server.");
      resultCard.style.display = "none";
    }
  });

  // Clear button
  clearBtn.addEventListener("click", () => {
    userText.value = "";
    resultCard.style.display = "none";
    warningMsg.style.display = "none";
    localStorage.removeItem("userText");
    localStorage.removeItem("lastResult");
  });

  // Example buttons
  if (exampleHumanBtn) {
    exampleHumanBtn.addEventListener("click", () => {
      const humanSamples = [
        "I am a student of Saranathan College of Engineering perusing a B.E degree in Computer Science and Engineering and I want to become a Data Scientist. I will become one.",
        "I felt sorrow for my uncle who had just met with an accident. I went to see him in the Hospital."
      ];
      const sampleText = humanSamples[Math.floor(Math.random() * humanSamples.length)];
      userText.value = sampleText;
      localStorage.setItem("userText", sampleText);
    });
  }

  if (exampleAiBtn) {
    exampleAiBtn.addEventListener("click", () => {
      const aiSamples = [
        "Today was incredible. The team finally launched the project we’ve been working on for months, and the feedback was overwhelmingly positive. Everyone was smiling, high-fiving, and celebrating the win. It felt like all the late nights were worth it.",
        "The quarterly report indicates a 3.2% increase in user engagement, primarily driven by mobile traffic. Conversion rates remained stable, while bounce rates showed a slight decline across key landing pages.",
        "Her reaction was not impulsive—it was calculated fury. The betrayal she endured had eroded every ounce of trust, replacing it with a cold, simmering resentment. She no longer sought reconciliation; she sought accountability."
      ];
      const sampleText = aiSamples[Math.floor(Math.random() * aiSamples.length)];
      userText.value = sampleText;
      localStorage.setItem("userText", sampleText);
    });
  }
});
