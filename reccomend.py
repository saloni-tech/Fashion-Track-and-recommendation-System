import cv2
import os
import time
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('model/body_shape_model.h5')

# Define body shape classes
classes = ['hourglass', 'rectangle', 'pear', 'apple', 'inverted_triangle']

# Helper function to predict body shape
def predict_body_shape(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize image to the input size expected by the model
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)  # Get model prediction
    return classes[np.argmax(prediction)]  # Return the body shape class with the highest probability

# Helper function to generate a unique output filename
def get_output_filename(image_path):
    timestamp = int(time.time())  # Use timestamp for unique filenames
    original_name = os.path.basename(image_path)
    name, ext = os.path.splitext(original_name)
    return f"{name}_{timestamp}{ext}"

# Main function to recommend an outfit
def recommend_outfit(image_path):
    body_shape = predict_body_shape(image_path)  # Predict body shape
    img = cv2.imread(image_path)  # Read the input image

    # Define outfit suggestions for each body shape
    suggestions = {
        'hourglass': ("dresses/hourglass.png"),
        'rectangle': ("dresses/rectangle.png"),
        'pear': ("dresses/pear.png"),
        'apple': ("dresses/apple.png"),
        'inverted_triangle': ("dresses/inverted_triangle.png")
    }

    # Get the label and dress image filename based on predicted body shape
    label, dress_image = suggestions.get(body_shape, ('Outfit', 'default.jpg'))

    # Add text to the image with the recommended outfit
    text = f"Recommended: {label}"
    cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the output image with the recommendation
    output_name = get_output_filename(image_path)
    output_path = os.path.join("static/output", output_name)
    cv2.imwrite(output_path, img)

    # Return the paths of the output image and recommended dress image
    dress_path = os.path.join("dresses", dress_image)  # Relative path for use in HTML
    return output_path, dress_path
