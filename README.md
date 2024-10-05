
# SMS Spam Classifier

This project is a **web-based SMS Spam Classification Application** built using **Flask**. The main goal of the application is to classify SMS messages as *Spam* or *Not Spam* using a machine learning model. The project was developed collaboratively by **Anand Kumar Dubey** and **Debmala Adhikari** from the **MSc Data Science** program.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [How to Use](#how-to-use)
6. [Model and Preprocessing](#model-and-preprocessing)
7. [Contributors](#contributors)
8. [Future Enhancements](#future-enhancements)

## Overview
The SMS Spam Classifier uses a machine learning model to classify SMS messages as either "Spam" or "Not Spam". The application is powered by a **Bernoulli Naive Bayes** classifier trained on a labeled dataset of SMS messages. The web interface is built using Flask, and the model is integrated to make predictions based on user inputs.

## Features
- **Interactive Web Application**: Users can input a message through a web form and get instant classification results.
- **Text Preprocessing**: The app performs text preprocessing (e.g., stopword removal, stemming) to clean input messages.
- **Machine Learning**: The classification is performed using a trained Bernoulli Naive Bayes model.
- **User-Friendly Interface**: Simple and clean design for ease of use.

## Project Structure
```
SMS-Spam-Classifier/
├── static/              # Static files such as CSS, images
├── templates/           # HTML templates
│   └── index.html       # Main page template for the web app
├── spam.csv             # Dataset used for training and testing
├── app.py               # Main Flask application
├── spam_classifier_model.pkl   # Serialized machine learning model
├── tfidf_vectorizer.pkl         # TF-IDF Vectorizer used for text transformation
├── README.md            # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/SMS-Spam-Classifier.git
   ```

2. Navigate to the project directory:
   ```bash
   cd SMS-Spam-Classifier
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include:

   ```
   Flask
   numpy
   pandas
   scikit-learn
   nltk
   joblib
   ```

4. Download NLTK stopwords and punkt tokenizer if not already present:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## How to Use
1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. You will see a text box where you can input an SMS message. After entering the message, click on the "Classify Message" button.

4. The result (Spam or Not Spam) will be displayed on the same page below the text box.

## Model and Preprocessing
- **Dataset**: The model is trained on the **SMS Spam Collection Dataset**, which is publicly available and consists of SMS messages labeled as either 'ham' (not spam) or 'spam'.
- **Text Preprocessing**: The preprocessing steps include:
  - Lowercasing the text
  - Tokenization using `nltk`
  - Removing special characters and stopwords
  - Stemming the tokens using the `PorterStemmer`
- **Model**: A Bernoulli Naive Bayes classifier is used to build the model, and the text data is vectorized using the **TF-IDF (Term Frequency-Inverse Document Frequency)** method.

## Contributors
This project was developed by:
- **Anand Kumar Dubey** - [LinkedIn]([https://www.linkedin.com/in/anand-dubey-27ba511b0/])
- **Debmala Adhikari**

Both contributors are final-year MSc Data Science students working together on this project to learn more about text classification and Flask web development.

## Future Enhancements
Here are some potential future features and improvements:
1. **Integration with SMS API**: Implementing a feature to send real-time SMS for spam detection.
2. **Deploying on the Cloud**: Hosting the application using a cloud service like AWS, Azure, or Heroku.
3. **Model Improvement**: Exploring other machine learning models such as SVM, Random Forest, or deep learning approaches.
4. **User Authentication**: Adding login and authentication for users to save and view previous classification results.

