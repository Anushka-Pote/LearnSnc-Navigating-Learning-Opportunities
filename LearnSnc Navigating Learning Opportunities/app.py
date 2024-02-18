# app.py

from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

app = Flask(__name__)
app.static_folder = 'static'

# Read data from CSV file
file_path = 'Ai&DS.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Map categorical features to numerical values
difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}

# Function to run K-means clustering and display results
def run_kmeans(subject, platform, difficulty, duration, rating):
    selected_subject_df = df[(df['Subject'] == subject) & (df['Platform'] == platform)]

    selected_subject_df['Difficulty'] = selected_subject_df['Difficulty'].map(difficulty_mapping)

    features = selected_subject_df[['Difficulty', 'Duration', 'Rating']]

    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    kmeans = KMeans(n_clusters=4, random_state=50)
    selected_subject_df['Cluster'] = kmeans.fit_predict(features_scaled)

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    selected_subject_df['PCA1'] = features_pca[:, 0]
    selected_subject_df['PCA2'] = features_pca[:, 1]

    user_input = pd.DataFrame({
        'Difficulty': [difficulty],
        'Duration': [duration],
        'Rating': [rating]
    })

    user_input['Difficulty'] = user_input['Difficulty'].map(difficulty_mapping)

    user_input_imputed = imputer.transform(user_input)
    user_scaled = scaler.transform(user_input_imputed)

    user_cluster = kmeans.predict(user_scaled)

    plt.figure(figsize=(10, 6))
    for cluster in selected_subject_df['Cluster'].unique():
        cluster_data = selected_subject_df[selected_subject_df['Cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=100, c='red',
                label='Centroids')
    plt.scatter(user_scaled[:, 0], user_scaled[:, 1], marker='*', s=100, c='green', label='User Input')
    plt.title(f'K-means Clustering of {subject} Courses on {platform} with User Input')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()

    # Save the plot to a BytesIO object
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_data = base64.b64encode(img_stream.read()).decode('utf-8')
    plt.close()

    return img_data, selected_subject_df[selected_subject_df['Cluster'] == user_cluster[0]]
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

questions = [
    {"question": "What does CSS stand for?", "options": ["Counter Strike: Source", "Cascading Style Sheets", "Computer Science", "Corrective Style Sheet"], "correct": 1},
    {"question": "What is the capital of France?", "options": ["Berlin", "Paris", "Madrid", "Rome"], "correct": 1},
    {"question": "What is the main purpose of HTML?", "options": ["Styling content", "Programming", "Structuring content", "Database management"], "correct": 2},
    {"question": "Which programming language is known for its simplicity and readability?", "options": ["Java", "C", "Python", "Ruby"], "correct": 2},
    {"question": "What is the result of 2 + 2 * 3?", "options": ["8", "10", "12", "14"], "correct": 3},]


@app.route("/", methods=['GET', 'POST'])
def index():
    img_data = None
    recommended_courses_df = None

    if request.method == 'POST':
        subject = request.form['subject']
        platform = request.form['platform']
        difficulty = request.form['difficulty']
        duration = int(request.form['duration'])
        rating = float(request.form['rating'])

        img_data, recommended_courses_df = run_kmeans(subject, platform, difficulty, duration, rating)

    return render_template('index.html', img_data=img_data, recommended_courses_df=recommended_courses_df)

@app.route("/Chatbot")
def home():
    return render_template("Chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/browse')
def browse():
    return render_template('browse.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/exam')
def index():
    if request.method == 'POST':
        score = 0
        for i, q in enumerate(questions):
            user_answer = request.form.get(f'subject{i}', '').strip().lower()
            correct_answer = q['correct'].lower()
            if user_answer == correct_answer:
                score += 1
        return render_template('exam.html', questions=questions, show_result=True, score=score, num_questions=len(questions))
    return render_template('exam.html', questions=questions)
if __name__ == '__main__':
    app.run(debug=True)
