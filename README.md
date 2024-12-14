

# Dating AI Adversary

## Overview
**Dating AI Adversary** is an advanced AI agent designed to intervene in dating conversations. The system employs sophisticated sentiment analysis, emotion detection, and dynamic question generation to stress-test relationships and facilitate truth revelation, compatibility exploration, and emotional depth evaluation.

This project integrates **Google GenerativeAI (Gemini)** and **Hugging Face Transformers** to analyze conversation sentiment, detect emotions, and generate strategically crafted questions based on contextual insights.

---

## Features
- **Automatic Speech Recogniton** : Added voice to text feature via recognition model
- **Sentiment Analysis**: Classifies recent messages as *Positive*, *Negative*, or *Neutral*.
- **Emotion Detection**: Extracts emotions (e.g., joy, anger, sadness) from conversations.
- **Toxicity Detection**: Flags harmful or toxic language and adjusts emotional scores accordingly.
- **Dynamic Question Generation**:
  - **Truth Revelation**: Probes unspoken truths.
  - **Compatibility Probe**: Assesses compatibility dimensions.
  - **Emotional Depth**: Challenges emotional connection and growth.
- **Advanced Privacy Handling**: Protects sensitive user data by anonymizing emails and phone numbers.
- **Multithreaded server** : allows both users to simultanaeously send messages 

---

## Tech Stack
- **Fastwhisper and pyaudio** for asr feature
- **Google GenerativeAI** (Gemini-1.5-flash) for question generation.
- **Hugging Face Transformers**:
  - Emotion Classification: `michellejieli/emotion_text_classifier`
  - Toxicity Detection: `unitary/unbiased-toxic-roberta`
  - Sentiment Analysis: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Python**: Core programming language.
- **NumPy**: For numerical operations.
- **Datetime**: To track intervention timestamps.
- **cpp based server (multithreaded)**

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kairveeehh/AI-dating-adversary-
   cd dating-ai-adversary
   ```

2. Install dependencies: can refer to https://github.com/kairveeehh/AI-dating-adversary-/requirements.txt
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your **Google GenerativeAI API Key** and **Hugging Face Token**:
   - Replace the placeholder `api_key` in the `main` function with your actual Google GenerativeAI API key.
   - For Hugging Face, generate a token from [Hugging Face](https://huggingface.co) and configure it during initialization.

---

## Usage

Run the AI agent using:
```bash
python main.py
```

### Example convos - 
 

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
4. **Abortion**: 
  - in the case everything is going right .   

---

## Privacy & Ethics
- **Anonymization**: Sensitive data such as emails and phone numbers are masked in preprocessing.
- **Ethical Boundaries**: The agent **never suggests termination** and focuses on constructive questioning.

---






## Contact
For queries or collaborations, contact [vkairvee@gmail.com].
