# ============================================
# Pneumonia Detection - Flask Backend
# Deployment Ready Version
# ============================================

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# ============================================
# Flask Application
# ============================================

app = Flask(__name__)


# ============================================
# Load Trained Pneumonia Detection Model
# ============================================

model = tf.keras.models.load_model("pneumonia_model.keras")


# ============================================
# Home + X-Ray Prediction
# ============================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        # Get uploaded X-ray image
        file = request.files.get("image")

        if file and file.filename:

            try:

                # Open uploaded image
                image = Image.open(file).convert("RGB")

                # Resize image to model input size
                image = image.resize((224, 224))

                # Convert image to NumPy array
                image_array = np.array(image) / 255.0

                # Add batch dimension
                image_array = np.expand_dims(image_array, axis=0)

                # AI prediction
                prediction = model.predict(
                    image_array,
                    verbose=0
                )[0][0]

                # Class decision
                if prediction > 0.5:
                    result = "PNEUMONIA"
                else:
                    result = "NORMAL"

            except Exception as e:

                result = "Error processing image"

                print("Error:", e)

    return render_template(
        "index.html",
        result=result
    )


# ============================================
# Run Flask Server - Deployment Ready
# ============================================

if __name__ == "__main__":

    # Get port from hosting server
    port = int(
        os.environ.get("PORT", 5000)
    )

    # Run Flask application
    app.run(
        host="0.0.0.0",
        port=port
    )
