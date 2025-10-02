import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model

def rgb_to_hsv(image):
   # Normalize pixel values to the range [0, 1]
    image_normalized = image.astype(np.float32) / 255.0

    # Extract R, G, B components
    R, G, B = image_normalized[:, :, 0], image_normalized[:, :, 1], image_normalized[:, :, 2]

    # Compute Value (V)
    V = np.max(image_normalized, axis=2)

    # Compute Saturation (S)
    denominator = np.where(V != 0, V, 1.0)
    S = (V - np.min(image_normalized, axis=2)) / denominator

    # Compute Hue (H)
    delta_R = (V - R) / (6 * denominator + 1e-10) + 1.0
    delta_G = (V - G) / (6 * denominator + 1e-10) + 1.0
    delta_B = (V - B) / (6 * denominator + 1e-10) + 1.0

    H = np.where(V == R, delta_B - delta_G, np.where(V == G, 2.0 + delta_R - delta_B, 4.0 + delta_G - delta_R))
    H = (H / 6.0) % 1.0

    return H * 360, S, V


def calculate_entropy(intensity_channel):
    # Calculate histogram of intensity values
    hist, _ = np.histogram(intensity_channel, bins=256, range=(0, 1))

    # Compute probability distribution
    prob_distribution = hist / np.sum(hist)

    # Remove zero probabilities to avoid NaN in the entropy calculation
    non_zero_probs = prob_distribution[prob_distribution > 0]

    # Calculate entropy
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    return entropy


def calculate_local_entropy_partial(intensity_channel, window_size=3):
    height, width = intensity_channel.shape

    # Calculate the number of non-overlapping blocks in height and width
    block_height = height // window_size
    block_width = width // window_size

    # Reshape the intensity channel to a 4D array with dimensions for block_height and block_width
    blocks = intensity_channel[:block_height * window_size, :block_width * window_size] \
        .reshape(block_height, window_size, block_width, window_size)

    # Calculate histogram for all blocks
    hist, _ = np.histogram(blocks, bins=256, range=(0, 1))

    # Compute probability distribution
    prob_distribution = hist / np.sum(hist)

    # Remove zero probabilities to avoid NaN in the entropy calculation
    non_zero_probs = np.where(prob_distribution > 0, prob_distribution, 1.0)

    # Calculate entropy for all blocks
    local_entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    return local_entropy



def calculate_rms_contrast(intensity_channel):
    # Calculate the standard deviation of the intensity channel
    std_intensity = np.std(intensity_channel)

    return std_intensity


def calculate_local_contrast(intensity_channel, window_size=3):
    height, width = intensity_channel.shape

    # Calculate the number of non-overlapping blocks in height and width
    block_height = height // window_size
    block_width = width // window_size

    # Reshape the intensity channel to a 4D array with dimensions for block_height and block_width
    blocks = intensity_channel[:block_height * window_size, :block_width * window_size] \
        .reshape(block_height, window_size, block_width, window_size)

    local_contrast = np.zeros((block_height, block_width))

    for i in range(block_height):
        for j in range(block_width):
            block = blocks[i, :, j, :]

            # Calculate standard deviation for the block
            local_contrast[i, j] = np.std(block)

    # Calculate the mean of local contrasts
    local_contrast_mean = np.mean(local_contrast)
            
    return local_contrast_mean



def normalize_value(value, min_val, max_val, new_min=1, new_max=5):
    normalized_value = ((value - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    return normalized_value


def process_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert BGR to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB to HSV
    H, S, V = rgb_to_hsv(image_rgb)

    # Calculate RMS contrast using the Intensity (V) component
    rms_contrast_value = calculate_rms_contrast(V)
    rms_contrast_normalized = normalize_value(rms_contrast_value, 0, 255)

    # Calculate the mean value of the S component
    mean_saturation = np.mean(S)
    mean_saturation_normalized = normalize_value(mean_saturation, 0, 1)

    # Calculate entropy based on the Intensity (V) component
    entropy_I = calculate_entropy(V)
    entropy_normalized = normalize_value(entropy_I, 0, -np.log2(1/256))

    # Calculate local entropy
    loc_ent = calculate_local_entropy_partial(V)
    loc_ent_normalized = normalize_value(loc_ent, 0, -np.log2(1/256))

    # Calculate local contrast
    local_contrast = calculate_local_contrast(V, 5)
    local_contrast_normalized = normalize_value(local_contrast, 0, 255)

    return [rms_contrast_normalized, entropy_normalized, local_contrast_normalized,loc_ent_normalized, mean_saturation_normalized]

def calculate_similarity(pArr, cArr):
  similarity = []
  for i in range(len(pArr)):
    if pArr[i] == cArr[i]:
      similarity.append(100)
    else:
      max_val = max(pArr[i], cArr[i])
      min_val = min(pArr[i], cArr[i])
      difference = max_val - min_val
      similarity.append((1 - difference/max_val) * 100)
  return similarity

def calculate_weights(similarity):
  total_similarity = sum(similarity)
  weights = [sim / total_similarity for sim in similarity]
  return weights

def calculate_final_value(cArr, weights):
  final_value = 0
  for i in range(len(cArr)):
    final_value += cArr[i] * weights[i]
  return final_value


# Create GUI
class ImageQualityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Quality Predictor")
        self.root.geometry("600x400")

        self.model = load_model('image_quality_model.keras')

        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.pack()

        self.btn_browse = tk.Button(self.root, text="Browse Image", command=self.browse_image)
        self.btn_browse.pack()

        self.lbl_results = tk.Label(self.root, text="")
        self.lbl_results.pack()

    def browse_image(self):
        # Clear previous results
        for widget in self.root.winfo_children():
            if isinstance(widget, (tk.Frame, tk.Label)):
                widget.destroy()

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            image = cv2.imread(file_path)
            image_resized = cv2.resize(image, (100, 100))
            image = np.expand_dims(image_resized, axis=0)
            image = image / 255.0

            # Predict quality
            prediction = self.model.predict(image)

            # Display image
            image_rgb = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            w, h = pil_image.size
            aspect_ratio = w / h

            # Resize the image to fit the canvas
            if aspect_ratio > 1:
                new_width = 500
                new_height = int(500 / aspect_ratio)
            else:
                new_width = int(500 * aspect_ratio)
                new_height = 500

            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            image_tk = ImageTk.PhotoImage(pil_image)
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            self.canvas.image = image_tk  # Keep a reference to prevent garbage collection

            # Process image for additional features
            calc_results = process_image(file_path)

            # Calculate similarity and weights
            similarity = calculate_similarity(prediction[0], calc_results)
            weights = calculate_weights(similarity)
            final_value = calculate_final_value(calc_results, weights)

            # Display results in columns
            results_frame = tk.Frame(self.root)
            results_frame.pack()

            # Predicted quality column
            predicted_quality_frame = tk.Frame(results_frame)
            predicted_quality_frame.pack(side=tk.LEFT, padx=10)

            lbl_predicted_quality = tk.Label(predicted_quality_frame, text="Predicted Quality:")
            lbl_predicted_quality.pack()

            for i, label in enumerate(['Contrast', 'Entropy', 'Local Contrast', 'Local Entropy', 'Saturation']):
                lbl_result = tk.Label(predicted_quality_frame, text=f"{label}: {prediction[0][i]}")
                lbl_result.pack(anchor=tk.W)

            # Calculated features column
            calculated_features_frame = tk.Frame(results_frame)
            calculated_features_frame.pack(side=tk.LEFT, padx=10)

            lbl_calculated_features = tk.Label(calculated_features_frame, text="Calculated Features:")
            lbl_calculated_features.pack()

            for i, label in enumerate(['RMS Contrast', 'Entropy', 'Local Contrast', 'Local Entropy', 'Mean Saturation']):
                lbl_result = tk.Label(calculated_features_frame, text=f"{label}: {calc_results[i]}")
                lbl_result.pack(anchor=tk.W)

            # Similarity percentages column
            similarity_frame = tk.Frame(results_frame)
            similarity_frame.pack(side=tk.LEFT, padx=10)

            lbl_similarity = tk.Label(similarity_frame, text="Similarity percentages:")
            lbl_similarity.pack()

            for i, label in enumerate(['Contrast', 'Entropy', 'Local Contrast', 'Local Entropy', 'Saturation']):
                lbl_result = tk.Label(similarity_frame, text=f"{label}: {similarity[i]}")
                lbl_result.pack(anchor=tk.W)

            # Weights column
            weights_frame = tk.Frame(results_frame)
            weights_frame.pack(side=tk.LEFT, padx=10)

            lbl_weights = tk.Label(weights_frame, text="Weights:")
            lbl_weights.pack()

            for i, label in enumerate(['Contrast', 'Entropy', 'Local Contrast', 'Local Entropy', 'Saturation']):
                lbl_result = tk.Label(weights_frame, text=f"{label}: {weights[i]}")
                lbl_result.pack(anchor=tk.W)

            # Final value
            lbl_final_value = tk.Label(self.root, text=f"Final value: {final_value}")
            lbl_final_value.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageQualityGUI(root)
    root.mainloop()
