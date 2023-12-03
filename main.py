import time

import tensorflow as tf
import tensorflow_datasets as tfds

from models import dog_cat_model
from keras import Sequential
from keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom


def dog_cat_train_augmented(model):
    (cat_train, cat_valid), info = tfds.load('cats_vs_dogs',
                                             split=['train[:80%]', 'train[80%:]'],
                                             with_info=True,
                                             as_supervised=True)

    def pre_process_image(image, label):
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        image = tf.image.resize(image, (128, 128))
        return image, label

    batch_size = 32
    shuffle_buffer_size = 1000

    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2),
        RandomZoom(0.2),
    ])

    train_data = cat_train.map(pre_process_image)
    train_data = train_data.map(lambda image, label: (data_augmentation(image), label))

    train_data = train_data.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_data = cat_valid.map(pre_process_image).batch(batch_size)

    t_start = time.time()
    model.fit(train_data, epochs=5, validation_data=validation_data,
              steps_per_epoch=len(cat_train) // batch_size,
              validation_steps=len(cat_valid) // batch_size, callbacks=None)
    print("Training done, dT:", time.time() - t_start)
    return model


model = dog_cat_model()
trained_model = dog_cat_train_augmented(model)
trained_model.save('dogs_cats_trained.h5')
