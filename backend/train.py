import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

DATA_PATH = "../data/fake_news_dataset.xls"


MODEL_DIR = "../model"
os.makedirs(MODEL_DIR, exist_ok=True)


# load dataset 
news_dataset = pd.read_csv(DATA_PATH)
news_dataset = news_dataset.fillna("")      



news_dataset['content'] = news_dataset['author'] + " " + news_dataset['title']+ " " + news_dataset['text']

X = news_dataset['content']
Y = news_dataset['label']


vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, token_pattern=r'[a-zA-Z]+')
X = vectorizer.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)


joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))

print(f"Model and vectorizer saved to: {MODEL_DIR}/")
