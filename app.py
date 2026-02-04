from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from camera import VideoCamera
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import os
import tempfile

app = Flask(__name__)

# Secret key for session management
app.secret_key = 'your_secret_key_here'

# Database setup for SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model for SQLAlchemy
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Initialize camera for body shape prediction
camera = VideoCamera()

# Load the trained body shape classification model
model = load_model("body_shape_model.h5")
print("Model input shape:", model.input_shape)  # Optional: for debugging

# List of body shape classes (must match your training order)
classes = ['hourglass', 'rectangle', 'pear', 'apple', 'inverted_triangle']

# ✅ Updated preprocessing to match model's input shape
def preprocess_image(frame):
    frame = cv2.resize(frame, (224, 224))  # Resized to match model input
    frame = img_to_array(frame) / 255.0
    return np.expand_dims(frame, axis=0)  # Shape becomes (1, 224, 224, 3)

# Home Route (Landing Page)
@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html', logged_in=True)
    return render_template('index.html', logged_in=False)

# Sign Up Route
@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    # Check if email or username already exists in the database
    user = User.query.filter((User.username == username) | (User.email == email)).first()
    if user:
        flash('Username or Email already exists!', 'error')
        return redirect(url_for('index'))
    
    # Hash the password and store user data in the database
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    flash('Account created successfully! Please log in.', 'success')
    return redirect(url_for('login'))

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username exists in the 'users' table
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# Video Feed Route for live camera stream
@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Prediction Route for body shape detection and dress recommendation
@app.route('/predict', methods=['POST'])
def predict():
    frame = camera.get_raw_frame()
    if frame is None:
        return jsonify({"error": "Camera capture failed"})

    # Save the captured image temporarily (optional for debugging or future use)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, frame)

    # Preprocess and predict
    img = preprocess_image(frame)
    prediction = model.predict(img)

    # Debugging: Print the prediction to check if it's giving valid probabilities
    print("Prediction output:", prediction)

    body_shape_class = np.argmax(prediction[0])
    predicted_shape = classes[body_shape_class]

    # Debugging: Print the predicted shape to ensure it's correct
    print("Predicted Body Shape:", predicted_shape)

    # Recommend dress images based on predicted shape
    dress_folder = os.path.join('static', 'dresses', predicted_shape)
    if not os.path.exists(dress_folder):
        print(f"No folder found for {predicted_shape} at {dress_folder}")
        return jsonify({
            "body_shape": predicted_shape,
            "recommendation_images": []
        })

    dress_images = [
        f"/static/dresses/{predicted_shape}/{fname}"
        for fname in os.listdir(dress_folder)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    return jsonify({
        "body_shape": predicted_shape,
        "recommendation_images": dress_images
    })

if __name__ == '__main__':
    # Create the database and tables if they do not exist
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)