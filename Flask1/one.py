from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# ----------------- Keep all your processing functions -----------------
def rgb_to_hsv(image):
    image_normalized = image.astype(np.float32) / 255.0
    R, G, B = image_normalized[:, :, 0], image_normalized[:, :, 1], image_normalized[:, :, 2]
    V = np.max(image_normalized, axis=2)
    denominator = np.where(V != 0, V, 1.0)
    S = (V - np.min(image_normalized, axis=2)) / denominator
    delta_R = (V - R) / (6 * denominator + 1e-10) + 1.0
    delta_G = (V - G) / (6 * denominator + 1e-10) + 1.0
    delta_B = (V - B) / (6 * denominator + 1e-10) + 1.0
    H = np.where(V == R, delta_B - delta_G, np.where(V == G, 2.0 + delta_R - delta_B, 4.0 + delta_G - delta_R))
    H = (H / 6.0) % 1.0
    return H * 360, S, V

def calculate_entropy(intensity_channel):
    hist, _ = np.histogram(intensity_channel, bins=256, range=(0, 1))
    prob_distribution = hist / np.sum(hist)
    non_zero_probs = prob_distribution[prob_distribution > 0]
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    return entropy

def calculate_local_entropy_partial(intensity_channel, window_size=3):
    height, width = intensity_channel.shape
    block_height, block_width = height // window_size, width // window_size
    blocks = intensity_channel[:block_height * window_size, :block_width * window_size].reshape(block_height, window_size, block_width, window_size)
    hist, _ = np.histogram(blocks, bins=256, range=(0, 1))
    prob_distribution = hist / np.sum(hist)
    non_zero_probs = np.where(prob_distribution > 0, prob_distribution, 1.0)
    local_entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    return local_entropy

def calculate_rms_contrast(intensity_channel):
    return np.std(intensity_channel)

def calculate_local_contrast(intensity_channel, window_size=3):
    height, width = intensity_channel.shape
    block_height, block_width = height // window_size, width // window_size
    blocks = intensity_channel[:block_height * window_size, :block_width * window_size].reshape(block_height, window_size, block_width, window_size)
    local_contrast = np.zeros((block_height, block_width))
    for i in range(block_height):
        for j in range(block_width):
            block = blocks[i, :, j, :]
            local_contrast[i, j] = np.std(block)
    return np.mean(local_contrast)

def normalize_value(value, min_val, max_val, new_min=1, new_max=5):
    return ((value - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, S, V = rgb_to_hsv(image_rgb)
    rms_contrast = normalize_value(calculate_rms_contrast(V), 0, 255)
    mean_saturation = normalize_value(np.mean(S), 0, 1)
    entropy_norm = normalize_value(calculate_entropy(V), 0, -np.log2(1/256))
    local_entropy_norm = normalize_value(calculate_local_entropy_partial(V), 0, -np.log2(1/256))
    local_contrast_norm = normalize_value(calculate_local_contrast(V, 5), 0, 255)
    return [rms_contrast, entropy_norm, local_contrast_norm, local_entropy_norm, mean_saturation]

def calculate_similarity(pArr, cArr):
    similarity = []
    for i in range(len(pArr)):
        if pArr[i] == cArr[i]:
            similarity.append(100)
        else:
            max_val, min_val = max(pArr[i], cArr[i]), min(pArr[i], cArr[i])
            similarity.append((1 - (max_val - min_val)/max_val) * 100)
    return similarity

def calculate_weights(similarity):
    total_similarity = sum(similarity)
    return [sim/total_similarity for sim in similarity]

def calculate_final_value(cArr, weights):
    return sum(cArr[i] * weights[i] for i in range(len(cArr)))

# ----------------- Flask App -----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

from keras.models import load_model

model = load_model("image_quality_model.h5")


@app.route("/", methods=["GET", "POST"])
def index():
    result_data = None
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Predict using model
            img = cv2.imread(file_path)
            img_resized = cv2.resize(img, (100, 100))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0
            prediction = model.predict(img_input)[0]

            # Process image features
            calc_results = process_image(file_path)
            similarity = calculate_similarity(prediction, calc_results)
            weights = calculate_weights(similarity)
            final_value = calculate_final_value(calc_results, weights)

            result_data = {
                "filename": file.filename,
                "prediction": prediction,
                "calc_results": calc_results,
                "similarity": similarity,
                "weights": weights,
                "final_value": final_value
            }

    return render_template("index.html", result=result_data)

if __name__ == "__main__":
    app.run(debug=True)
