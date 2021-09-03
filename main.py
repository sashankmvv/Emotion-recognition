from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from utils import *
from model import model
from tensorflow.keras.optimizers import Adam

init_lr = 3e-4
epochs = 50
batch_size = 32
metrics = {
    'emotion_output': 'accuracy',
    'age_output': 'accuracy',
    'race_output': 'accuracy',
    'gender_output': 'accuracy'
}
ages_train, emotions_train, genders_train, races_train = seperatecategory(y_train)
ages_test, emotions_test, genders_test, races_test = seperatecategory(y_test)

train_gen = generate_images(X_train, emotions_train,
                            genders_train, races_train, ages_train, 32, True)
valid_gen = generate_images(X_test, emotions_test,
                            genders_test, races_test, ages_test, 32, True)
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