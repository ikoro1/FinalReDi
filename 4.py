import os
from flask import Flask, request, render_template
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

# Argument Parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='resnet50', choices=[
    'resnet50', 'vgg16', 'convnextbase'])
args = vars(parser.parse_args())

# Model Dictionary
models_dict = {
    'resnet50': tf.keras.applications.resnet50.ResNet50(weights='imagenet'),
    'vgg16': tf.keras.applications.vgg16.VGG16(weights='imagenet'),
    'convnextbase': tf.keras.applications.convnext.ConvNeXtBase(weights='imagenet'),
}

def process_image(file_path):
    orig_image = plt.imread(file_path)
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.imagenet_utils.preprocess_input(image)

    # Convert the image to base64
    image_buffer = BytesIO()
    plt.imsave(image_buffer, orig_image, format='png')
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

    results = []
    for model_name, model in models_dict.items():
        predictions = model.predict(image)
        processed_preds = tf.keras.applications.imagenet_utils.decode_predictions(predictions)
        result = {
            'model_name': model_name,
            'class_name': processed_preds[0][0][1],
            'confidence': processed_preds[0][0][2] * 100,
        }
        results.append(result)

    return image_base64, results

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file part'
        else:
            file = request.files['file']

            if file.filename == '':
                error = 'No selected file'
            elif file:
                upload_folder = 'uploads'
                if not os.path.exists(upload_folder):
                    os.makedirs(upload_folder)
                file_path = os.path.join(upload_folder, file.filename)
                file.save(file_path)

                image_base64, results = process_image(file_path)

                result = {'image_base64': image_base64, 'results': results}

    return render_template('index4.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True, port=5003)
