from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image, ImageOps
import os
import base64


app = Flask(__name__)
MODEL_PATH = ('./models/', './digit_model.h5')
MODEL = load_model(os.path.join(*MODEL_PATH))
CORS(app)



def preprocess_image(image_b64):
    image_b64 = image_b64.split(',')[-1]  
    missing_padding = len(image_b64) % 4
    if missing_padding:
        image_b64 += '=' * (4 - missing_padding)

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Grayscale

    image = ImageOps.invert(image)

    image = image.point(lambda x: 0 if x < 100 else 255, '1')
    image = image.convert('L')

    image = ImageOps.fit(image, (28, 28), method=Image.LANCZOS)
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
    return img_array



@app.route("/")
def home():
    return "API is running!"


@app.route("/predict_handwritten_number", methods=["POST"])
def predict():
    if 'image' not in request.json:
        return jsonify({"error": " No image Provided"})

    image = request.json['image']
    img_array = preprocess_image(image)
    predictions = MODEL.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return jsonify({"predicted_class": int(predicted_class)})

@app.route("/health", methods=["GET"])
def health_check():
    try:
        MODEL.predict(np.zeros((1, 28, 28, 1))) 
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)