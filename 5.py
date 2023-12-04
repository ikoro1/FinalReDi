from flask import Flask, request, render_template
import os
import base64
from io import BytesIO
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# disable message   This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dictionary of models (mapping in-code model name to actual TensorFlow model)
models_dict = {'vgg16': tf.keras.applications.vgg16.VGG16(weights='imagenet'),
    'resnet50': tf.keras.applications.resnet50.ResNet50(weights='imagenet'),
    'mobilenet': tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet'),
    'convnextbase': tf.keras.applications.convnext.ConvNeXtBase(weights='imagenet'),
    }
img_formats_dict = {'vgg16': (224, 224),
    'resnet50': (224, 224),
    'mobilenet': (224, 224),
    'convnextbase': (224, 224),
    }


def preprocess_image(file_path, target_size):
    '''Converts original image into a suitable format for TensorFlow'''

    image_tf = tf.keras.preprocessing.image.load_img(file_path, target_size=target_size)
    image_tf = np.expand_dims(image_tf, axis=0)
    image_tf = tf.keras.applications.imagenet_utils.preprocess_input(image_tf)
    return image_tf


def image_to_base64(file_path):
    '''Converts original image to base64'''

    # Read the original image
    orig_image = plt.imread(file_path)
    # Convert the image to base64
    image_buffer = BytesIO()
    plt.imsave(image_buffer, orig_image, format='png')
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    return image_base64


def classify_image(image_tf, tf_model, model_name):
    ''' Classifies the image and get the top-3 probable classes of the picture with probabilities '''

    predictions = tf_model.predict(image_tf)
    processed_preds = tf.keras.applications.imagenet_utils.decode_predictions(predictions)
    prediction_result = {'model_name': model_name,
        'class_name1': processed_preds[0][0][1],
        'confidence1': processed_preds[0][0][2] * 100,
        'class_name2': processed_preds[0][1][1],
        'confidence2': processed_preds[0][1][2] * 100,
        'class_name3': processed_preds[0][2][1],
        'confidence3': processed_preds[0][2][2] * 100,
        }
    return prediction_result


def process_image(file_path, models_dict, img_formats_dict):
    ''' Performs image classification for each model in the dictionary
        Returns probable classes of the picture with its probabilities
    '''

    results = []
    for model_name, tf_model in models_dict.items():
        # Get correct image size for current model
        target_size = img_formats_dict[model_name]
        # Get image for TensorFlow model
        image_tf = preprocess_image(file_path, target_size = target_size)
        # Get results from TensorFlow model
        result = classify_image(image_tf = image_tf, tf_model = tf_model, model_name = model_name)
        # Append new element to models results
        results.append(result)
    return results


#
### Test without flask
#

#filename = 'iff.jpeg'
#file_path = os.path.join('C:\\Users\\User\\Documents\\Study_Python\\ReDi\\Python-2023\\Final_Project\\input', filename)
#image_base64, results = process_image(file_path, models_dict, img_formats_dict)
#print(results)


# Setting up Flask
app = Flask(__name__)
# Routing to main (index) page
@app.route('/', methods=['GET', 'POST'])

def index():
    ''' Index page '''

    result = None
    error = None

    # Handling POST requests
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

                # Performing classification
                results = process_image(file_path, models_dict, img_formats_dict)
                # Getting original image for page rendering
                image_base64 = image_to_base64(file_path)
                
                result = {'image_base64': image_base64, 'results': results}

    return render_template('test_with_flask.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True, port=5003)

#
### ???
#

# file extension checker
# error handler
# several pages on web-site
