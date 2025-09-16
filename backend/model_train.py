import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv")
X = df['tweet']
y = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)  # 1 = hate/offensive, 0 = not hate

# Train model
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

model.fit(X, y)
joblib.dump(model, 'hate_speech_model.pkl')
print("✅ Model saved as hate_speech_model.pkl")
