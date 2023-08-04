from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from utils.utils import load_data
from pathlib import Path

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(36, activation='softmax'))  # 10 digits + 26 letters = 36 classes

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


X_train, y_train, X_test, y_test = load_data()

model = build_model()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=256)

with open(Path(__file__).parent / 'evals/cnn_sequentional_eval.txt', 'w') as f:

    f.write('Train Accuracy:\n')
    f.write('\n'.join(str(a) for a in history.history['accuracy']))
    f.write('\n\n')
    f.write('Validation Accuracy:\n')
    f.write('\n'.join(str(a) for a in history.history['val_accuracy']))
    f.write('\n\n')
    f.write('Train Loss:\n')
    f.write('\n'.join(str(l) for l in history.history['loss']))
    f.write('\n\n')
    f.write('Validation Loss:\n')
    f.write('\n'.join(str(l) for l in history.history['val_loss']))

model.save(Path(__file__).parent.parent /'trained_models/cnn_sequentional.h5')
