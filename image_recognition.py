from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os


def dog_cat_predict(model, image_file):
    label_names = ["cat", "dog"]

    img = load_img(image_file, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = int(np.round(predictions[0][0]))
    predicted_class = label_names[predicted_class_index]

    confidence_percentage = predictions[0][0] * 100  # Переводим вероятность в проценты

    return predicted_class, confidence_percentage


def predict_images_in_folder(model, folder_path):
    files = os.listdir(folder_path)
    files.sort()

    for file_name in files:
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            image_path = os.path.join(folder_path, file_name)
            prediction, confidence = dog_cat_predict(model, image_path)
            print(f'File: {file_name}, Prediction: {prediction}, Confidence: {confidence:.2f}%')


model_path = 'dogs_cats_trained.h5'
loaded_model = load_model(model_path)

folder_with_images = 'data'
predict_images_in_folder(loaded_model, folder_with_images)
