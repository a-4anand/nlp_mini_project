# Import necessary libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess the dataset
df = pd.read_csv('spam.csv', encoding='latin1')

# Drop unnecessary columns
df = df[['v1', 'v2']]
df.columns = ['SMSlabel', 'TEXT']

# Remove duplicates and missing values
df = df.drop_duplicates(keep='first')
df.dropna(inplace=True)

# Encode labels (ham = 0, spam = 1)
df['SMSlabel'] = df['SMSlabel'].map({'ham': 0, 'spam': 1})

# Text Preprocessing Function
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Convert to lower case
    tokens = nltk.word_tokenize(text)  # Tokenize text
    tokens = [re.sub(r'[^a-z]', '', token) for token in tokens]  # Remove special characters
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(token) for token in tokens]  # Stem the tokens
    return ' '.join(tokens)

df['TRANSFORMED_TEXT'] = df['TEXT'].apply(preprocess_text)

# Convert text into numerical data using TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['TRANSFORMED_TEXT']).toarray()
y = df['SMSlabel'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train Bernoulli Naive Bayes model
model = BernoulliNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Flask app setup
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Define the main route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        processed_message = preprocess_text(message)
        vectorized_message = tfidf.transform([processed_message]).toarray()
        prediction = model.predict(vectorized_message)
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template('index.html', prediction_text=f'The message is: {result}')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
