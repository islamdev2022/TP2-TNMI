from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app, supports_credentials=True)
print("CORS has been enabled")
def bruit_poivre_et_sel(image, prob):
    noisy_image = np.copy(image)
    for i in range(noisy_image.shape[0]):
        for j in range(noisy_image.shape[1]):
            rand = random.random()
            if rand < prob:
                noisy_image[i][j] = 0  # Pepper
            elif rand > 1 - prob:
                noisy_image[i][j] = 255  # Salt
    return noisy_image

@app.route('/', methods=['GET'])
def home():
    return "Hello, World!", 200

@app.route("/noise", methods=["POST"])
def upload_noise():
    if "image" in request.files:
        image_file = request.files["image"]
        conversion_mode = request.form["conversion_mode"]
        prob = float(request.form["value"])  # Convert to float
        
        # Read the image and convert to a NumPy array
        image = Image.open(image_file).convert("L")  # Convert to grayscale if needed
        image_np = np.array(image)

        if conversion_mode == 'poivre et sel':
            noisy_image = bruit_poivre_et_sel(image_np, prob)
            
            # Convert back to an image
            noisy_image_pil = Image.fromarray(noisy_image)
            
            # Save or return the image as needed
            buf = io.BytesIO()
            noisy_image_pil.save(buf, format="PNG")
            buf.seek(0)
            return (buf.getvalue(), 200, {'Content-Type': 'image/png'})
        else:
            return "Invalid conversion mode", 400
    else:
        return "No image received", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
