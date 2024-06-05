from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

positive_review = "This hotel exceeded our expectations. The rooms were spacious and clean, the staff was incredibly friendly and helpful, and the breakfast was delicious with a wide variety of options. The location was perfect, just a short walk from the beach. We will definitely stay here again!"
negative_review = "This was the worst hotel experience I have ever had. The rooms were dirty and smelled awful, the staff was rude and unhelpful, and the breakfast was terrible with very few choices. The location was noisy and far from any attractions. I will never stay here again!"

positive_scores = sia.polarity_scores(positive_review)
negative_scores = sia.polarity_scores(negative_review)

print("Positive review scores:", positive_scores)
print("Negative review scores:", negative_scores)

import text2emotion as te

positive_emotions = te.get_emotion(positive_review)
negative_emotions = te.get_emotion(negative_review)

print("Positive review emotions:", positive_emotions)
print("Negative review emotions:", negative_emotions)

# Wyniki są raczej zgodne z oczekiwaniami, jednak nie w 100%