import os
from flask import Flask, request, render_template
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.src.applications import ConvNeXtBase

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            upload_folder = 'uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            orig_image = plt.imread(file_path)
            image = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
            image = np.expand_dims(image, axis=0)
            image = tf.keras.applications.imagenet_utils.preprocess_input(image)

            plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

            for i, (model_name, model) in enumerate(models_dict.items()):
                plt.subplot(1, 3, i + 1)

                predictions = model.predict(image)
                processed_preds = tf.keras.applications.imagenet_utils.decode_predictions(predictions)

                plt.imshow(orig_image)
                plt.title(f"{model_name}: {processed_preds[0][0][1]}, {processed_preds[0][0][2] * 100:.3f}")
                plt.axis('off')

            plt.show()

            return render_template('index.html', result='Images and predictions displayed.')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
