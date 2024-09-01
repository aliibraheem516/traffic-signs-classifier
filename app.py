from flask import Flask, render_template, request,url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the TensorFlow model
model = tf.keras.models.load_model('saved_model/traffic_sign_classifier.keras')

# Dictionary mapping class indices to their corresponding traffic sign labels
class_labels = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

def preprocess_image(image):
    """Preprocess the uploaded image to match the model input requirements."""
    # Resize the image to 32x32
    image = image.resize((32, 32))

    # Convert the image to grayscale
    image = image.convert('L')

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the image data
    image_array = image_array / 128.0

    # Reshape the array to match model input (1, 32, 32, 1)
    image_array = image_array.reshape(1, 32, 32, 1)

    return image_array

def clear_upload_folder():
    """Delete all files in the upload folder."""
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    for f in files:
        os.remove(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image_url = None
    if request.method == 'POST':
        # Clear the upload folder before saving the new file
        clear_upload_folder()

        # Check if an image is part of the POST request
        if 'file' not in request.files:
            return render_template('index.html', error='No file part in the request')

        file = request.files['file']

        # Check if the user actually selected a file
        if file.filename == '':
            return render_template('index.html', error='No file selected for uploading')

        try:
            # Save the uploaded image to a static folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Get the URL of the uploaded image to display it in the template
            uploaded_image_url = url_for('static', filename=f'uploads/{file.filename}')

            # Open the image file
            image = Image.open(image_path)

            # Preprocess the image
            processed_image = preprocess_image(image)

            # Make predictions
            predictions = model.predict(processed_image)

            # Get the class with the highest probability
            predicted_class = np.argmax(predictions, axis=1)[0]

            # Map the predicted class to the corresponding traffic sign label
            predicted_label = class_labels[predicted_class]

            # Return the prediction and display the uploaded image
            return render_template('index.html', prediction=f'Predicted class: {predicted_label}', uploaded_image_url=uploaded_image_url)

        except Exception as e:
            return render_template('index.html', error=str(e), uploaded_image_url=uploaded_image_url)

    return render_template('index.html', uploaded_image_url=uploaded_image_url)

if __name__ == '__main__':
    app.run(debug=True)
