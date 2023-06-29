from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, add
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from utils.utils import load_data

def residual_block(x, filters, conv_num=2, activation='relu'):
    # Shortcut
    s = Conv2D(filters, (1, 1), padding='same')(x)

    for _ in range(conv_num):
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

    x = add([s, x])
    return Activation(activation)(x)


def build_model():
    inputs = Input(shape=(28, 28, 1))

    x = residual_block(inputs, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 128, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 256, 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    outputs = Dense(36, activation='softmax')(x)  # 10 digits + 26 letters = 36 classes

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model

X_train, y_train, X_test, y_test = load_data()

model = build_model()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=256)

with open(Path(__file__).parent / 'evals/resnet_eval.txt', 'w') as f:
    # Write the accuracy for each epoch
    f.write('Train Accuracy:\n')
    f.write('\n'.join(str(a) for a in history.history['accuracy']))
    f.write('\n\n')

    f.write('Validation Accuracy:\n')
    f.write('\n'.join(str(a) for a in history.history['val_accuracy']))
    f.write('\n\n')

    # Write the loss for each epoch
    f.write('Train Loss:\n')
    f.write('\n'.join(str(l) for l in history.history['loss']))
    f.write('\n\n')

    f.write('Validation Loss:\n')
    f.write('\n'.join(str(l) for l in history.history['val_loss']))

# Save model
model.save(Path(__file__).parent.parent / 'trained_models/resnet.h5')
