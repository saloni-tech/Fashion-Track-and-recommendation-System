
```markdown
#  AI-Based Fashion Recommendation System

An intelligent fashion recommendation web application that detects a user's **body shape using deep learning** and provides **personalized outfit suggestions**. The system uses image processing and a trained neural network model to classify body types and recommend suitable fashion styles.

---

## 🚀 Features

- 📸 Real-time body shape detection using camera input
- 🧠 Deep Learning model for body shape classification
- 👚 Personalized fashion recommendations based on body type
- 🌐 Web-based interface using Flask
- 🔐 User authentication (Login & Registration)
- 📂 Dataset-based model training
- ⚡ Fast and interactive UI

---

## 🧩 Body Shapes Supported

- Apple
- Pear
- Hourglass
- Rectangle
- Inverted Triangle

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- Bootstrap

### Backend
- Python
- Flask
- Flask-SQLAlchemy

### AI / ML
- TensorFlow / Keras
- OpenCV
- NumPy

### Database
- SQLite

---

## 📁 Project Structure

```

recommanded-system-main/
│
├── app.py                     # Main Flask application
├── camera.py                  # Camera handling & video stream
├── body_shape_model.h5        # Trained deep learning model
├── dataset/                   # Training dataset (body shapes)
├── static/                    # CSS, images, JS
├── templates/                 # HTML templates
└── README.md                  # Project documentation

````

---

## ⚙️ How It Works

1. User logs into the system
2. Camera captures the body image
3. Image is preprocessed using OpenCV
4. Deep Learning model predicts the body shape
5. System recommends fashion styles accordingly

---

## ▶️ Installation & Setup

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/ai-fashion-recommendation-system.git
````

### Step 2: Navigate to project directory

```bash
cd recommanded-system-main
```

### Step 3: Install required dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
python app.py
```

### Step 5: Open in browser

```
http://127.0.0.1:5000/
```

---

## 📊 Model Details

* Model Type: CNN (Convolutional Neural Network)
* Framework: Keras (TensorFlow backend)
* Input: Body image
* Output: Body shape category

---

## 📌 Use Cases

* Online fashion platforms
* Virtual try-on systems
* Personalized shopping assistants
* Fashion recommendation engines

---

## 🔮 Future Enhancements

* 🔄 Real-time AR try-on feature
* 🧍 Full body measurement extraction
* 🛍️ E-commerce integration
* 📱 Mobile app version
* 🤖 Improved recommendation logic using user feedback

---



