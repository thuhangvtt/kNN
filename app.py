import os
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from scipy.signal import find_peaks
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model đã train sẵn
with open("knn_model_manh.pkl", "rb") as f:
    knn_model: KNeighborsClassifier = pickle.load(f)

# ==== Tiền xử lý ảnh cho MNIST ====
def preprocess_image(path):
    img_color = cv2.imread(path)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Histogram để quyết định đảo ngược
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_smoothed = cv2.GaussianBlur(hist, (9, 1), 0).flatten()
    peaks, _ = find_peaks(hist_smoothed, distance=20, height=100)
    if len(peaks) >= 2:
        p1, p2 = sorted(peaks, key=lambda x: hist_smoothed[x], reverse=True)[:2]
        low_mean, high_mean = sorted([p1, p2])
        if high_mean - low_mean < 40 or high_mean < low_mean:
            gray = 255 - gray

    gray = cv2.fastNlMeansDenoising(gray, None, h=20)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX))

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)
    binary = cv2.medianBlur(binary, 3)

    if np.mean(binary) < 25:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
            binary = mask

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8), path

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[max(y - 4, 0):y + h + 4, max(x - 4, 0):x + w + 4]
    resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_CUBIC)

    M = cv2.moments(resized)
    cx = int(M["m10"] / M["m00"]) if M["m00"] else 10
    cy = int(M["m01"] / M["m00"]) if M["m00"] else 10
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset, y_offset = 14 - cx, 14 - cy
    for y in range(20):
        for x in range(20):
            ny, nx = y + y_offset, x + x_offset
            if 0 <= ny < 28 and 0 <= nx < 28:
                canvas[ny, nx] = resized[y, x]

    final_img = cv2.GaussianBlur(canvas, (3, 3), 0)
    _, final_img = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)

    processed_path = path.replace(".", "_processed.")
    cv2.imwrite(processed_path, final_img)
    return final_img, processed_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            processed_img, processed_path = preprocess_image(image_path)
            img_flat = processed_img.reshape(1, -1) / 255.0

            probs = knn_model.predict_proba(img_flat)[0]
            pred = np.argmax(probs)
            confidence = round(probs[pred] * 100, 2)

            return render_template('index.html',
                                   prediction=pred,
                                   confidence=confidence,
                                   image_url='/' + image_path,
                                   processed_url='/' + processed_path)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
