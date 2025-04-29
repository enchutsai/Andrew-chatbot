import joblib
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

classifier = joblib.load("trained_model/sentiment_mlpclassifier.pkl")
tfidf_vect = joblib.load("trained_model/sentiment_tfidf_vectorizer.pkl")
encoder = joblib.load("trained_model/sentiment_label_encoder.pkl")

def predict_sentiment(text):
    vec = tfidf_vect.transform([text])
    pred = classifier.predict(vec)[0]
    label = encoder.inverse_transform([pred])[0]
    return label