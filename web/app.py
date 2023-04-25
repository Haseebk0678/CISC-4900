# import libraries
from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image


model = load_model('./best_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
input_shape = (100,100)

# define a route for the image upload
@app.route('/predict', methods=['POST'])
def predict():
    # get the uploaded image
    image = request.files['image'].read()
    image = np.fromstring(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, input_shape)

    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Resize the image to the correct dimensions (if necessary)
    image_array = np.resize(image_array, (1, 100, 100))
    
    # Normalize the image data
    image_array = image_array / 255.0
    # use the CNN model to predict if a mask is present
    prediction = model.predict(image)[0][0]
    if prediction > 0.5:
        result = 'Mask is present'
    else:
        result = 'Mask is not present'
    
    # render the prediction result to the user
    return render_template('index.html', prediction=result)

# run the Flask app
if __name__ == '__main__':
    app.run(debug=True)