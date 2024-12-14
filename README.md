

# Dating AI Adversary

## Overview
**Dating AI Adversary** is an advanced AI agent designed to intervene in dating conversations. The system employs sophisticated sentiment analysis, emotion detection, and dynamic question generation to stress-test relationships and facilitate truth revelation, compatibility exploration, and emotional depth evaluation.

This project integrates **Google GenerativeAI (Gemini)** and **Hugging Face Transformers** to analyze conversation sentiment, detect emotions, and generate strategically crafted questions based on contextual insights.

---

## Features
- **Sentiment Analysis**: Classifies recent messages as *Positive*, *Negative*, or *Neutral*.
- **Emotion Detection**: Extracts emotions (e.g., joy, anger, sadness) from conversations.
- **Toxicity Detection**: Flags harmful or toxic language and adjusts emotional scores accordingly.
- **Dynamic Question Generation**:
  - **Truth Revelation**: Probes unspoken truths.
  - **Compatibility Probe**: Assesses compatibility dimensions.
  - **Emotional Depth**: Challenges emotional connection and growth.
- **Advanced Privacy Handling**: Protects sensitive user data by anonymizing emails and phone numbers.

---

## Tech Stack
- **Google GenerativeAI** (Gemini-1.5-flash) for question generation.
- **Hugging Face Transformers**:
  - Emotion Classification: `michellejieli/emotion_text_classifier`
  - Toxicity Detection: `unitary/unbiased-toxic-roberta`
  - Sentiment Analysis: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Python**: Core programming language.
- **NumPy**: For numerical operations.
- **Datetime**: To track intervention timestamps.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/dating-ai-adversary.git
   cd dating-ai-adversary
   ```

2. Install dependencies:
   ```bash
   pip install google-generativeai transformers numpy
   ```

3. Configure your **Google GenerativeAI API Key** and **Hugging Face Token**:
   - Replace the placeholder `api_key` in the `main` function with your actual Google GenerativeAI API key.
   - For Hugging Face, generate a token from [Hugging Face](https://huggingface.co) and configure it during initialization.

---

## Usage

Run the AI agent using:
```bash
python dating_ai_adversary.py
```

### Example Conversation Input:
```python
conversation_history = [
    {"sender": "Alex", "content": "You're the most useless partner I've ever had. You never do anything right!"},
    {"sender": "Riley", "content": "Stop yelling at me! You're always so critical and mean."},
    {"sender": "Alex", "content": "Maybe if you weren't such a complete failure, I wouldn't have to criticize you constantly."},
    {"sender": "Riley", "content": "I hate you. You make me feel worthless and terrible about myself."},
    {"sender": "Alex", "content": "Good. Maybe then you'll actually try to improve for once in your pathetic life."}
]
```

### Sample Output:
```
Intervention Details:
{
  "timestamp": "2024-07-16T10:45:32.123Z",
  "strategy": "truth_revelation",
  "sentiment_analysis": {
    "POSITIVE": 0,
    "NEGATIVE": 4,
    "NEUTRAL": 1
  },
  "emotion_scores": {
    "anger": 0.75,
    "sadness": 0.60
  },
  "toxicity_score": 0.85,
  "dynamic_question": "Is there something deeper causing resentment between you two that you've never openly discussed?"
}
```

---

## Code Structure

- **`DatingAIAdversary` Class**:
  - `analyze_sentiment`: Sentiment analysis for recent messages.
  - `analyze_emotions`: Emotion and toxicity analysis with normalization.
  - `select_strategy`: Chooses a strategy based on emotions and toxicity.
  - `intervene`: Generates a dynamic intervention question.
- **Prompts**: Predefined strategies for *truth revelation*, *compatibility probing*, and *emotional depth exploration*.

---

## Strategies

1. **Truth Revelation**:
   - Focuses on hidden relationship truths to challenge the couple’s communication.
2. **Compatibility Probe**:
   - Illuminates relational compatibility dimensions.
3. **Emotional Depth**:
   - Tests emotional strength and vulnerability.

---

## Privacy & Ethics
- **Anonymization**: Sensitive data such as emails and phone numbers are masked in preprocessing.
- **Ethical Boundaries**: The agent **never suggests termination** and focuses on constructive questioning.

---

## Future Enhancements
- Integrate real-time conversation monitoring.
- Expand emotion models for more granular classifications.
- Include multilingual support for global accessibility.

---

## Contribution
Contributions are welcome! To propose improvements:
1. Fork the repository.
2. Create a new branch.
3. Submit a pull request.

---

## License
This project is licensed under the **MIT License**.

---

## Contact
For queries or collaborations, contact [vkairvee@gmail.com].
