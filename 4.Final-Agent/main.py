import google.generativeai as genai
from transformers import pipeline
import os
import json
import re
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

class DatingAIAdversary:
    def __init__(self,
                 api_key: str,
                 model_name: str = "gemini-1.5-flash",
                 huggingface_token: str = None,
                 huggingface_emotion_model: str = "michellejieli/emotion_text_classifier",
                 huggingface_toxicity_model: str = "unitary/unbiased-toxic-roberta"):
        """
        Enhanced Dating AI Adversary with advanced sentiment and emotion analysis.

        :param api_key: Google GenerativeAI API key
        :param model_name: Google AI model to use
        :param huggingface_token: Hugging Face authentication token
        :param huggingface_emotion_model: Emotion classification model
        :param huggingface_toxicity_model: Toxicity detection model
        """
        genai.configure(api_key=api_key)
        self.google_model_name = model_name

        if huggingface_token:
            os.environ['HUGGINGFACE_TOKEN'] = huggingface_token
            from huggingface_hub import login
            login(token=huggingface_token)

        self.emotion_analyzer = pipeline("text-classification", model=huggingface_emotion_model)
        self.toxicity_analyzer = pipeline("text-classification", model=huggingface_toxicity_model)
        self.sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

        self.strategy_prompts = {
            "truth_revelation": {
                "core_prompt": """Relationship Truth-Revealing intervention:
the two people in the conversation are dating and you have to intervene them , refer both of the people in the question or one if you feel so based on their
 conversation to question the strength of their relationship, and try to stress-test the relationship by asking questions that invoke
  truth revelation and focus on testing the strength of the relationship:



Desired Outcome:
Generate a single short , precisely crafted question that will:
- Reveal hidden relationship truths based on the context given
- Encourages stress testing of the relatioship strength but do not initiate termination """,
                "generation_instruction": """Based on the comprehensive intervention protocol, generate a single short  deeply insightful question that will challenge the couple to confront their unspoken truths between them ."""
            },
            "compatibility_probe": {
                "core_prompt": """Relationship Compatibility Exploration Protocol:the two people in the conversation are dating and you have to intervene them , refer both of the people in the question or one if you feel so based on their
 conversation to question the strength of their relationship, and try to stress-test the relationship by asking questions that invoke a stress on the compatibility between them address them both



Desired Outcome:
Craft a single, precisely formulated question that will:
- Illuminate compatibility nuances
- Foster mutual understanding
- Enable strategic relationship recalibration""",
                "generation_instruction": """Leveraging the comprehensive compatibility exploration protocol, generate a single, strategically designed question that will help the couple deeply understand their relational compatibility across multiple dimensions."""
            },
            "emotional_depth": {
                "core_prompt": """Emotional Depth and Vulnerability Exploration Protocol:the two people in the conversation are dating and you have to intervene them , refer both of the people in the question or one if you feel so based on their
 conversation to question the strength of their relationship, and try to stress-test the relationship by asking questions that invoke emotional strength testing and test their vulnerabilities in emotional aspect



Desired Outcome:
Generate a single, short meticulously crafted question or statement that will:
- question the emotional strength of the relationship
- Facilitate mutual emotional growth""",
                "generation_instruction": """Using the comprehensive emotional depth exploration protocol, generate a single, precisely calibrated question that will test the couple based on emotional strength and vulnerabilities """
            }
        }

        # Advanced intervention tracking
        self.intervention_history = []

    def _preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing to protect privacy and standardize input.

        :param text: Input text to preprocess
        :return: Cleaned and anonymized text
        """
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text)

        return ' '.join(text.lower().split())

    def analyze_emotions(self, conversation: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Enhanced emotion analysis with more comprehensive insights.

        :param conversation: List of conversation messages
        :return: Emotion scores with advanced processing
        """
        combined_text = " ".join([self._preprocess_text(msg['content']) for msg in conversation[-5:]])

        emotion_results = self.emotion_analyzer(combined_text)

        emotion_scores = {}
        for result in emotion_results:
            label = result["label"]
            score = result["score"]
            emotion_scores[label] = emotion_scores.get(label, 0) + score

        toxicity_results = self.toxicity_analyzer(combined_text)
        toxicity_score = max([result['score'] for result in toxicity_results])

        if toxicity_score > 0.5:
            for key in ['joy', 'trust']:
                if key in emotion_scores:
                    emotion_scores[key] *= (1 - toxicity_score)

        # Normalize scores
        total_messages = len(conversation[-5:])
        normalized_emotions = {emotion: score / total_messages for emotion, score in emotion_scores.items()}

        return {
            "emotions": normalized_emotions,
            "toxicity_score": toxicity_score
        }

    def analyze_sentiment(self, conversation: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Analyze sentiment across recent conversation messages.

        :param conversation: List of conversation messages
        :return: Sentiment count dictionary
        """
        sentiment_counts = {
            "POSITIVE": 0,
            "NEGATIVE": 0,
            "NEUTRAL": 0
        }

        for msg in conversation[-5:]:  # Analyze last 5 messages
            sentiment_result = self.sentiment_analyzer(msg['content'])[0]
            # Handle potential lowercase labels and unexpected labels:
            label = sentiment_result['label'].upper()  # Convert to uppercase
            if label not in sentiment_counts:
                label = "NEUTRAL"  # Default to neutral for unknown labels
            sentiment_counts[label] += 1

        return sentiment_counts

    def select_strategy(self, emotion_analysis: Dict[str, Any]) -> str:
        """
        Enhanced strategy selection with nuanced decision-making.

        :param emotion_analysis: Emotion analysis results from conversation
        :return: Selected strategy
        """
        emotions = emotion_analysis["emotions"]
        toxicity_score = emotion_analysis["toxicity_score"]

        # Advanced strategy selection logic
        if toxicity_score > 0.7:
            return "truth_revelation"
        elif emotions.get("anger", 0) > 0.5 or emotions.get("disgust", 0) > 0.5:
            return "truth_revelation"
        elif emotions.get("joy", 0) > 0.5 and emotions.get("trust", 0) > 0.5:
            return "compatibility_probe"
        else:
            return "emotional_depth"

    def _generate_contextual_insight(self, conversation: List[Dict[str, str]]) -> str:
        """
        Generate additional contextual insights for deeper analysis.

        :param conversation: Conversation history
        :return: Contextual insights string
        """
        # Analyze conversation length and complexity
        message_lengths = [len(msg['content']) for msg in conversation[-5:]]
        avg_length = np.mean(message_lengths)

        return f"Conversation Complexity: Average message length {avg_length:.2f} chars"

    def intervene(self, conversation: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Enhanced intervention method with sentiment and emotion analysis.

        :param conversation: Conversation history
        :return: Intervention details
        """
        # Analyze sentiment
        sentiment_analysis = self.analyze_sentiment(conversation)

        # Analyze emotions and get toxicity score
        emotion_analysis = self.analyze_emotions(conversation)
        emotions = emotion_analysis["emotions"]
        toxicity_score = emotion_analysis["toxicity_score"]

        # Check for persistent negative sentiment or high toxicity
        if sentiment_analysis.get("NEGATIVE", 0) > 3 or toxicity_score > 0.6:
            # Use a more critical strategy if negative sentiment is high
            strategy_key = "truth_revelation"
        else:
            # Select strategy based on emotions
            strategy_key = self.select_strategy(emotion_analysis)

        # Prepare conversation context
        context = "\n".join([f"{msg['sender']}: {msg['content']}" for msg in conversation[-5:]])

        # Select the strategy's prompt details
        strategy_prompt = self.strategy_prompts[strategy_key]

        # Use Google GenerativeAI to generate a nuanced question
        response = genai.GenerativeModel(self.google_model_name).generate_content(
       f"Relationship Intervention Context:\n{context}\n\n"
       f"the two people in the conversation are dating and you are a AI agent whose task is to  intervene them and act  as a relationship stress-tester in dating conversations. You will periodically intervene in conversations between dating couples to test the strength of their connection through statement and questions(should be short and not tooo long) based over the strategy and emotions and context provided to you , focus on the emotions more while phrasing the question or statement and then follow the strategy given, Never NEVER EVER  suggest the termination  and also keep your ethical boundaries .\n\n"
       f"{strategy_prompt['core_prompt']}\n\n"
       f"{strategy_prompt['generation_instruction']}"
        )

        # Track intervention
        intervention = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_key,
            "sentiment_analysis": sentiment_analysis,
            "emotion_scores": emotions,
            "toxicity_score": toxicity_score,
            "dynamic_question": response.text.strip(),
        }
        self.intervention_history.append(intervention)

        return intervention

def main():
    """
    Demonstration of the enhanced Dating AI Adversary.
    """
    # Replace with your actual API key
    api_key = "AIzaSyCf4AWfWKXWdxvKrM4OPzelqSJbeN72Hs0"

    # Initialize the Dating AI Adversary
    agent = DatingAIAdversary(api_key=api_key)

    # Sample conversation
    conversation_history = [
    {"sender": "Nina", "content": "I don’t think I can go through with this presentation. I’m terrified."},
    {"sender": "Eli", "content": "Hey, you’ve got this! Remember how hard you prepared."},
    {"sender": "Nina", "content": "What if I mess up? Everyone’s going to laugh at me."} 
      ]




    # Perform intervention
    intervention = agent.intervene(conversation_history)

    # Print detailed intervention details
    print("Intervention Details:")
    print(json.dumps(intervention, indent=2))

    # Detailed breakdown of analysis
    print("\nDetailed Analysis:")
    print(f"Sentiment Analysis: {intervention['sentiment_analysis']}")
    print("\nEmotion Scores:")
    for emotion, score in intervention['emotion_scores'].items():
        print(f"{emotion}: {score:.4f}")
    print(f"\nToxicity Score: {intervention['toxicity_score']:.4f}")

    print("\nVeritas:", intervention["dynamic_question"])

if __name__ == "__main__":
    main()