from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model

nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# SQLite Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# Load the chatbot model and data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r"C:\Users\Ananya\Desktop\major_prjct\intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def allowed_file(filename):
    """Check if the uploaded file is allowed based on the extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Create a bag of words representation for the input sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the class of the input sentence."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Get a random response for the predicted intent."""
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tags'][0] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Route for home page
@app.route('/')
def index():
    return render_template('home.html')

# Login/Signup page route
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/dash')
def dash_page():
    return render_template('dash.html')

@app.route('/bot')
def bot_page():
    return render_template('bot.html')

# File upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle image file upload."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        
        # If user does not select a file, browser also
        # submits an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Secure the file name and save it to the upload folder
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File uploaded successfully!', 'success')
            return redirect(url_for('report_page', filename=filename))

    return render_template('report.html')

# Route to display uploaded file
@app.route('/report/<filename>')
def report_page(filename):
    """Render the report page with the uploaded image."""
    file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('report.html', filename=filename, file_url=file_url)

# Signup route
@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    # Check if email already exists
    user = User.query.filter_by(email=email).first()
    if user:
        flash('Email address already exists', 'error')
        return redirect(url_for('login_page'))

    # Hash password
    hashed_password = generate_password_hash(password, method='sha256')
    
    # Create new user
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    flash('Account created successfully', 'success')
    return redirect(url_for('login_page'))

# Login route
@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    # Query user by email
    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        flash('Invalid email or password', 'error')
        return redirect(url_for('login_page'))

    session['user_id'] = user.id
    session['username'] = user.username

    flash('Login successful', 'success')
    return redirect(url_for('dashboard'))

# Dashboard (after login)
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login to access this page', 'error')
        return redirect(url_for('login_page'))
    return render_template('dashboard.html', username=session['username'])

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

# Chatbot interaction route
@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from the user."""
    message = request.json['message']  # Expecting 'message' key
    ints = predict_class(message)
    res = get_response(ints, intents)
    return jsonify(res)  # Return response as JSON

if __name__ == "__main__":
    app.run(debug=True)
