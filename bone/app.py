from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import os
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO(r"sriram\bone\best (2) (1).pt")  # Replace with correct path if needed

UPLOAD_FOLDER = r"sriram\bone\static\uploads"
RESULT_FOLDER = r"sriram\bone\static\results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def transform_image(image):
    """Convert PIL image to numpy array"""
    image = image.convert('RGB')
    return np.array(image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    image = Image.open(image_path)
    transformed_image = transform_image(image)

    results = model.predict(transformed_image, conf=0.25)

    # Save prediction results
    result_image_filenames = []
    for i, result in enumerate(results):
        im_array = result.plot()
        im = Image.fromarray(im_array)
        result_filename = f'result_{i}.jpg'
        result_image_path = os.path.join(RESULT_FOLDER, result_filename)
        im.save(result_image_path)
        result_image_filenames.append(result_filename)

    return render_template('result.html',
                           image_filename=filename,
                           result_image_filenames=result_image_filenames)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
