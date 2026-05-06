import os
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# App config

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------
# Model parameters
# -----------------------
IMAGE_SIZE = 128
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
NUM_CLASSES = len(CLASS_NAMES)

# -----------------------
# Rebuild EXACT model
# -----------------------
base_model = VGG16(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last 3 layers (same as training)
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    Flatten(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation="softmax")
])

# Build model before loading weights
model.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

# Load weights
model.load_weights("model.h5")
print("Model loaded successfully")

# -----------------------
# Image preprocessing
# -----------------------
def preprocess_image(path):
    img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None
    tumor_status = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = preprocess_image(image_path)
            preds = model.predict(img)[0]

            class_id = int(np.argmax(preds))
            confidence = round(float(preds[class_id]) * 100, 2)

            prediction = CLASS_NAMES[class_id]

            if prediction == "no_tumor":
                tumor_status = "No Tumor Detected"
            else:
                tumor_status = "Tumor Detected"

            image_url = image_path

    return render_template(
        "index.html",
        prediction=prediction,
        tumor_status=tumor_status,
        confidence=confidence,
        image_url=image_url
    )

# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
