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
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_names[predicted_class_index]

    return predicted_class


model = load_model('dogs_cats.h5')
folder_path = 'data'
files = os.listdir(folder_path)

for file_name in files:
    if file_name.startswith('data') and file_name.endswith('.png') or file_name.endswith('jpg'):
        image_path = os.path.join(folder_path, file_name)
        prediction = dog_cat_predict(model, image_path)
        print(f'File: {file_name}, Prediction: {prediction}')
