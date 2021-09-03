from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import cv2
# from google.colab.patches import cv2_imshow

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Lambda
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
from urllib.request import urlretrieve
IMG_WIDTH, IMG_HEIGHT = 50, 50

def make_hidden_layers(inputs):
    x = Conv2D(
        filters=64,  # number of filters
        kernel_size=(5, 5),  # the filter size or the kernel size
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal'
    )(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(
        filters=64,
        kernel_size=(5, 5),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.6)(x)
    x = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(
        128,
        activation='elu',
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    return x


input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
inputs = Input(shape=input_shape)
x = make_hidden_layers(inputs)

emotion_output = Dense(
    7,
    activation='softmax',
    name='emotion_output'
)(x)
gender_output = Dense(
    3,
    activation='softmax',
    name='gender_output'
)(x)
race_output = Dense(
    3,
    activation='softmax',
    name='race_output'
)(x)

age_output = Dense(
    5,
    activation='softmax',
    name='age_output'
)(x)
model = Model(
    inputs=inputs,
    outputs=[emotion_output, gender_output, race_output, age_output],
)
init_lr = 3e-4
epochs = 50
batch_size = 32

metrics = {
    'emotion_output': 'accuracy',
    'age_output': 'accuracy',
    'race_output': 'accuracy',
    'gender_output': 'accuracy'
}

model.compile(
    optimizer=Adam(learning_rate=init_lr, decay=init_lr / epochs),
    loss=[

        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    ],
    loss_weights=[2, 0.1, 1.5,  4, ],
    metrics=metrics
)

early_stopping = EarlyStopping(
    monitor='val_emotion_output_accuracy',
    min_delta=0.00005,
    patience=10,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_emotion_output_accuracy',
    factor=0.2,  # set to 0.1 or 0.2
    patience=5,  # set to 5-10
    min_lr=1e-6,  # might want to increase this, originally was 1e-7
    verbose=1,  # keep this as 1
)

callbacks = [
    early_stopping,
    lr_scheduler,
]

history = model.fit_generator(train_gen,
                              steps_per_epoch=len(y_train)//batch_size,
                              epochs=epochs,
                              callbacks=callbacks,
                              validation_data=valid_gen,
                              validation_steps=len(y_test)//batch_size)
