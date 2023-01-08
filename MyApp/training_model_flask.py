#Import required libraries

import base64
import os
import pickle
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template, redirect, url_for, flash
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shutil


PYTHONUNBUFFERED = "anything_her"

app = Flask(__name__)
app.secret_key = 'abababab123456'



app.config['UPLOAD_FOLDER'] = r'upload/'
upload_dir = app.config['UPLOAD_FOLDER']
app.config['MODEL_FOLDER'] = r'model/'
model_dir = app.config['MODEL_FOLDER']

label_encoder_name = 'label_encoder.pkl'
model_name = 'model.h5'
def load_custom_dataset(data_dir):
    # Load the image filenames and labels
    filenames = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for file in os.listdir(label_dir):
            filenames.append(os.path.join(label_dir, file))
            labels.append(label)

    # Convert the labels to integers
    num_classes = len(set(labels))
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    # Save the label encoder object to a file

    with open(os.path.join(model_dir, label_encoder_name), 'wb') as f:
        pickle.dump(label_encoder, f)

    # Preprocess the dataset
    image_size = 299
    images = []
    one_hot_labels = []
    for filename, label in zip(filenames, labels):
        # Load and decode the image
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, channels=3)

        # Resize the image to a fixed size
        image = tf.image.resize(image, (image_size, image_size))

        # Normalize the pixel values to the range [0, 1]
        image = image / 255.0

        # Convert the label to one-hot encoding
        label = tf.one_hot(label, num_classes)

        images.append(image)
        one_hot_labels.append(label)

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(images, one_hot_labels, test_size=0.2, shuffle=True)

    # Return the training and test data as tuples
    return (x_train, y_train), (x_test, y_test), num_classes

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the Start Training button was clicked

        if 'start_training' in request.form:
            os.makedirs(model_dir, exist_ok=True)
            # Get the data directory from the form
            if 'files' in request.files:
                # Get the uploaded folder
                uploaded_folder = request.files.getlist("files")
                for file in uploaded_folder:
                    # Check if the selected item is a folder
                    if file.filename == '':
                        # Walk through the selected folder and its subfolders
                        for root, dirs, files in os.walk(file.path):
                            # Recreate the folder structure on the server
                            server_folder = os.path.join(upload_dir, root[len(file.path) + 1:])
                            os.makedirs(server_folder, exist_ok=True)
                            # Upload the files in the current folder
                            for file in files:
                                file_path = os.path.join(root, file)
                                with open(file_path, 'rb') as f:
                                    file_content = f.read()
                                # Save the file to the server
                                with open(os.path.join(server_folder, file), 'wb') as f:
                                    f.write(file_content)
                    else:
                        # Get the file's directory and filename
                        directory, filename = os.path.split(file.filename)
                        # Construct the absolute path to the file on the server
                        server_file_path = os.path.join(upload_dir, directory, filename)
                        # Create the directory structure if it does not exist
                        os.makedirs(os.path.dirname(server_file_path), exist_ok=True)
                        # Save the file to the server
                        file.save(server_file_path)

                # Load the custom dataset
                data_dir, _ = os.path.split(directory)
                data_dir = os.path.join(upload_dir, data_dir)


                # Load the custom dataset
                (x_train, y_train), (x_test, y_test), num_classes = load_custom_dataset(data_dir)

                # Delete the temporary directory
                shutil.rmtree(upload_dir)

                # Load the Inception v3 model and remove the top layer
                base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)

                # Add a new top layer
                x = base_model.output
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dense(1024, activation='relu')(x)
                predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

                # Create the new model
                model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

                # Freeze the base model layers
                for layer in base_model.layers:
                    layer.trainable = False

                # Compile the model with a learning rate of 0.001 and a loss function of categorical crossentropy
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                              metrics=['accuracy'])

                # Normalize the pixel values in the training data to the range [0, 1]
                x_train = np.array(x_train)
                x_test = np.array(x_test)
                y_train = np.array(y_train)
                y_test = np.array(y_test)

                # Fit the model to the training data
                history = model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data=(x_test, y_test))

                # Save the model
                model.save(os.path.join(model_dir, model_name))
                flash('Training completed successfully!', 'success')

        if 'predict' in request.form:
            if 'files' in request.files:
                uploaded_folder = request.files.getlist("files")
                predictions = []
                count = 0
                image_data_uris = []
                for file in uploaded_folder:
                    # Check if the selected item is a folder
                    if file.filename != '':
                        # Get the file's directory and filename
                        count += 1
                        image_content = file.read()
                        image = tf.image.decode_image(image_content, channels=3)
                        image = tf.image.resize(image, (299, 299))
                        image = image / 255.0
                        image = np.expand_dims(image, axis=0)
                        directory, filename = os.path.split(file.filename)
                        if 'Original Inception V3 Model' in request.form.getlist('checkbox'):
                            # Load the original model
                            model = tf.keras.applications.InceptionV3(weights='imagenet')
                            predictionss = model.predict(image)
                            # Decode the predictions
                            predicted_classes = tf.keras.applications.inception_v3.decode_predictions(predictionss, top=1)
                            label_name = str(predicted_classes[0][0][1]).title()
                            confidence = predicted_classes[0][0][2]
                            confidence = round(confidence, 2)
                            predictions.append((count, filename, label_name, confidence))
                        else:
                            if not os.path.isdir(model_dir):
                                alert = 'You need to train model first!! Model file does not exist'
                                return render_template('upload.html', alert=alert)
                            elif len(os.listdir(model_dir)) == 0:
                                alert = 'You need to train model first!! Model file does not exist'
                                return render_template('upload.html', alert=alert)
                            # Load the label encoder object from a file

                            with open(os.path.join(model_dir, label_encoder_name), 'rb') as f:
                                label_encoder = pickle.load(f)
                        # Load the trained model
                            model = tf.keras.models.load_model(os.path.join(model_dir, model_name))
                            # Make a prediction
                            prediction = model.predict(image)
                            class_index = np.argmax(prediction[0])
                            confidence = prediction[0][class_index]
                            confidence = round(confidence, 2)
                            # Get the label names from the label encoder
                            label_names = label_encoder.inverse_transform([class_index])
                            label_name = str(label_names[0]).title()
                            predictions.append((count, filename, label_names[0], confidence))
                        # Create a file-like object from the image data
                        image_file = BytesIO(image_content)
                        # Load the image using PIL
                        # Load the image using PIL
                        image = Image.open(image_file)
                        image = image.resize(size=(200,200))
                        # Create a draw object
                        draw = ImageDraw.Draw(image)

                        # Choose a font and font size
                        font = ImageFont.truetype('arial.ttf', 16)

                        # Get the size of the image
                        width, height = image.size

                        # Calculate the position of the text
                        text_x = 10
                        text_y = 10
                        # Draw the text on the image
                        draw.text((text_x, text_y), label_name, font=font, fill=(255, 255, 255))

                        text_width, text_height = draw.textsize(str(confidence), font=font)
                        # Calculate the position of the text
                        text_x = width - text_width - 10
                        text_y = 10

                        # Draw the text on the image
                        draw.text((text_x, text_y), str(confidence), font=font, fill=(255, 255, 255))

                        text_width, text_height = draw.textsize(str(filename), font=font)
                        # Calculate the position of the text
                        text_x = 10
                        text_y = height - text_height - 10

                        # Draw the text on the image
                        draw.text((text_x, text_y), str(filename), font=font, fill=(255, 255, 255))

                        # Save the image to a buffer
                        buffer = BytesIO()
                        image.save(buffer, format='jpeg')
                        # Encode the image data as a data URI
                        image_data = buffer.getvalue()
                        image_data_uri = base64.b64encode(image_data).decode("utf-8")
                        image_data_uri = f"data:image/jpeg;base64,{image_data_uri}"

                        # Append the data URI to the list of image data URIs
                        image_data_uris.append(image_data_uri)


                return render_template('upload.html', predictions=predictions, image_data_uris=image_data_uris)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

