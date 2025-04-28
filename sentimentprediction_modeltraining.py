import gdown
import pandas as pd
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import joblib

# 下載與處理資料
gdown.download("https://drive.google.com/uc?id=1jbb0HQ9oTZkRNK055AvXD3ya8_ZLtN9o", "datasets/Amazon_review.csv", quiet=False)
df = pd.read_csv("datasets/Amazon_review.csv", delimiter=',', header=None)
df.columns = ['text', 'label']

# 分割資料
train_x, holdout_x, train_y, holdout_y = model_selection.train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# TF-IDF 向量化
tfidf_vect = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf_vect.fit(train_x)
xtrain = tfidf_vect.transform(train_x)
xholdout = tfidf_vect.transform(holdout_x)

# Label Encoding
encoder = preprocessing.LabelEncoder()
train_y_en = encoder.fit_transform(train_y)
holdout_y_en = encoder.transform(holdout_y)

# 建立模型
classifier = MLPClassifier(hidden_layer_sizes=(2,1), max_iter=100, random_state=1)
classifier.fit(xtrain, train_y_en)

joblib.dump(classifier, "trained_model/sentiment_mlpclassifier.pkl")
joblib.dump(tfidf_vect, "trained_model/sentiment_tfidf_vectorizer.pkl")
joblib.dump(encoder, "trained_model/sentiment_label_encoder.pkl")