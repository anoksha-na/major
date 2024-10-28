import tensorflow as tf
from mri_brain_test import predict_brain_tumor
from alz_test import predict_image_class
from ham_test import predict_brain_condition
from chest_ct_test import predict_chest_class
from covid_test import predict_image
from datetime import datetime
import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify, send_from_directory
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LOGGED_IN_PATIENTS'] = 0

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
model = load_model('chatbot_model.keras')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

#Database definition
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS doctors 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, email TEXT, password TEXT, doctor_reg_id TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS patients 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, email TEXT, password TEXT, pat_reg_id TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS scans 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, scan_type TEXT, patient_name TEXT, patient_age INTEGER, scan_no TEXT, prediction TEXT, timestamp DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS counts 
             (id INTEGER PRIMARY KEY AUTOINCREMENT, mri_scans_count INTEGER, ct_scans_count INTEGER, logged_in_patients_count INTEGER)''')
    c.execute("INSERT INTO counts (mri_scans_count, ct_scans_count, logged_in_patients_count) VALUES (0, 0, 0)")
    conn.commit()
    conn.close()

# Register a new doctor
def register_new_doctor(username, email, password, doctor_reg_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO doctors (username, email, password, doctor_reg_id) VALUES (?, ?, ?, ?)",
              (username, email, password, doctor_reg_id))
    conn.commit()
    conn.close()

# Register a new patient
def register_new_patient(username, email, password, pat_reg_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO patients (username, email, password, pat_reg_id) VALUES (?, ?, ?, ?)",
              (username, email, password, pat_reg_id))
    conn.commit()
    conn.close()

# Verify user login credentials for doctors
def authenticate_doctor(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM doctors WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Verify user login credentials for patients
def authenticate_patient(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Check if user exists (for signup page)
def check_user_exists(username, table):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute(f"SELECT * FROM {table} WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
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
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag in i['tags']:  # Check if the predicted intent matches any of the tags in the list
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No image found', 400
        image = request.files['image']
        if image.filename == '':
            return 'No image selected', 400
        if image:
            # Save the uploaded image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            # Extract scan type and category from form data of doc_dash.html
            scan_type = request.form['scanType']
            scan_category = request.form['scanCategory']

            # Extract patient details from form data
            patient_name = request.form['patientName']
            patient_age = request.form['age']
            scan_no = request.form['patientScanNo']

            # Determine the appropriate prediction function based on scan type and category
            if scan_type == "MRI":
                if scan_category == "Brain haemorrhage":
                    prediction = predict_brain_condition(image_path, 'brain_heamorrhage_model.keras')
                elif scan_category == "Brain Tumor":
                    prediction = predict_brain_tumor(image_path, 'mri_brain_train.keras')
                elif scan_category == "Alzheimer's":
                    prediction = predict_image_class(image_path, 'alzheimers_model.keras')
                else:
                    return 'Unsupported scan category', 400
            elif scan_type == "CT":

                if scan_category == "Chest CT":
                    prediction = predict_chest_class(image_path, 'chest_ct_model.h5')
                elif scan_category == "COVID-19":
                    model = tf.keras.models.load_model('covid_model.h5')
                    prediction = predict_image(image_path, model)    
                else:
                    return 'Unsupported scan category', 400
            else:
                return 'Unsupported scan type', 400
            
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO scans (scan_type, patient_name, patient_age, scan_no, prediction, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (scan_type, patient_name, patient_age, scan_no, prediction, datetime.now()))  
            conn.commit()
            conn.close()
            update_counts(scan_type)
            return redirect(url_for('result', prediction=prediction, scan_type=scan_type, scan_category=scan_category,
                                    patient_name=patient_name, patient_age=patient_age, scan_no=scan_no, image_filename=image.filename))
    elif request.method == 'GET':
        return render_template('doc_dash.html')
    
@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    scan_type = request.args.get('scan_type')
    scan_category = request.args.get('scan_category')
    patient_name = request.args.get('patient_name')
    patient_age = request.args.get('patient_age')
    scan_no = request.args.get('scan_no')
    image_filename = request.args.get('image_filename')
    return render_template('result.html', prediction=prediction, scan_type=scan_type, scan_category=scan_category,
                           patient_name=patient_name, patient_age=patient_age, scan_no=scan_no, image_filename=image_filename)

@app.route('/login_doctor', methods=['POST'])
def login_doctor():
    username = request.form['username']
    password = request.form['password']
    user = authenticate_doctor(username, password)
    if user:
        session['username'] = username
        flash('Logged in successfully!', 'success')
        return redirect(url_for('doc_dash'))  
    else:
        flash('Invalid username or password. Please try again or sign up.', 'error')
        return redirect(url_for('login_doc'))

@app.route('/login_patient', methods=['POST'])
def login_patient():
    username = request.form['username']
    password = request.form['password']
    user = authenticate_patient(username, password)
    if user:
        session['username'] = username
        app.config['LOGGED_IN_PATIENTS'] += 1  # Increment the count of logged-in patients
        flash('Logged in successfully!', 'success')
        return redirect(url_for('pat_dash'))
    else:
        flash('Invalid username or password. Please try again or sign up.', 'error')
        return redirect(url_for('login_pat'))

@app.route('/register_doctor', methods=['POST'])
def register_doctor():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    doctor_reg_id = request.form['doctor_reg_id']

    # Check if user already exists
    if check_user_exists(username, 'doctors'):
        flash('Username already exists. Please choose another username.', 'error')
        return redirect(url_for('login_doc'))

    register_new_doctor(username, email, password, doctor_reg_id)
    flash('Doctor registration successful! Please log in.', 'success')
    return redirect(url_for('login_doc'))  

@app.route('/register_patient', methods=['POST'])
def register_patient():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    pat_reg_id = request.form['pat_reg_id']

    # Check if user already exists
    if check_user_exists(username, 'patients'):
        flash('Username already exists. Please choose another username.', 'error')
        return redirect(url_for('login_pat'))

    register_new_patient(username, email, password, pat_reg_id)
    flash('Patient registration successful! Please log in.', 'success')
    return redirect(url_for('login_pat'))  

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('welcome.html', username=session['username'])
    else:
        flash('You need to log in first.', 'error')
        return redirect(url_for('home'))

@app.route('/dashboard1')
def dashboard1():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    else:
        flash('You need to log in first.', 'error')
        return redirect(url_for('home'))   

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route("/login_pat")
def login_pat():
    return render_template('loginpat.html')

@app.route("/login_doc")
def login_doc():
    return render_template('logindoc.html')

@app.route("/doc_dash")
def doc_dash():
    return render_template('doc_dash.html')

@app.route('/pat_dash')
def pat_dash():
    return render_template('pat_dash.html')

@app.route('/pat_rep')
def pat_rep():
    return render_template('pat_rep.html')

@app.route('/pat_rep_data', methods=['POST'])
def pat_rep_data():
    scan_no = request.form['scan_no']
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM scans WHERE scan_no=?", (scan_no,))
    scan_data = c.fetchone()
    conn.close()

    if scan_data:
        prediction = scan_data[5]
        return jsonify({'scan_data': scan_data, 'prediction': prediction})
    else:
        flash('Scan number not found. Please enter a valid scan number.', 'error')
        return redirect(url_for('pat_rep'))

def update_counts(scan_type):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    if scan_type == "MRI":
        c.execute("UPDATE counts SET mri_scans_count = mri_scans_count + 1")
    elif scan_type == "CT":
        c.execute("UPDATE counts SET ct_scans_count = ct_scans_count + 1")
    conn.commit()
    conn.close()

@app.route('/get_counts')
def get_counts():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute("SELECT * FROM counts")
    counts = c.fetchone()

    conn.close()

    return jsonify({
        'mriScansCount': counts[1],
        'ctScansCount': counts[2],
        'loggedInPatientsCount': app.config['LOGGED_IN_PATIENTS']
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chatbot_page')
def chatbot_page():
    return render_template('index.html',response="", user_input="")

@app.route('/index_chat', methods=['POST', 'GET'])
def index_chat():
    data = request.get_json()
    message = data.get('question')
    intent = predict_class(message)
    res = get_response(intent, intents)
    return jsonify({'response': res})

@app.route('/view_all_scans')
def view_all_scans():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM scans")
    scans = c.fetchall()
    conn.close()
    return render_template('view_all_scans.html', scans=scans)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', debug=True)