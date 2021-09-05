from copy import Error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Lambda
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.models import load_model
import gdown
import os

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


FER_model_id = "1-CxM2xlDpl4AeO9sXY2f3lkDe6YV63yu"
RAFDB_model_id = "1-5MG7jEHADCv3bc5tF3H_fQNJPl4QxEb"


def get_model(pretrained=False, dataset_name=None):
    if pretrained:
        output_model = f"{dataset_name}_model.h5"
        if not os.path.exists(output_model):
            print(f"Downloading model pretrained on {dataset_name} ...")
            if dataset_name == "FER":
                model_URL = f"https://drive.google.com/uc?id={FER_model_id}"
            elif dataset_name == "RAFDB":
                model_URL = f"https://drive.google.com/uc?id={RAFDB_model_id}"
            else:
                raise Error(
                    f"dataset_name should be FER or RAFDB not {dataset_name} when pretrained=True")

            gdown.download(model_URL, output_model, quiet=False)
        model = load_model(output_model)
    else:
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

        outputs = [emotion_output, gender_output, race_output, age_output]
        model = Model(
            inputs=inputs,
            outputs=outputs,
        )

    return model
