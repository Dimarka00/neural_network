import time

import tensorflow as tf
import tensorflow_datasets as tfds

from models import dog_cat_model


def dog_cat_train(model):
    splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']
    (cat_train, cat_valid, cat_test), info = tfds.load('cats_vs_dogs',
                                                       split=splits,
                                                       with_info=True,
                                                       as_supervised=True)

    def pre_process_image(image, label):
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        image = tf.image.resize(image, (128, 128))
        return image, label

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    train_batch = cat_train.map(pre_process_image).shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(BATCH_SIZE)
    validation_batch = cat_valid.map(pre_process_image).repeat().batch(BATCH_SIZE)

    t_start = time.time()
    model.fit(train_batch, steps_per_epoch=4000, epochs=2,
              validation_data=validation_batch,
              validation_steps=10,
              callbacks=None)
    print("Training done, dT:", time.time() - t_start)


model = dog_cat_model()
dog_cat_train(model)
model.save('dogs_cats.h5')
