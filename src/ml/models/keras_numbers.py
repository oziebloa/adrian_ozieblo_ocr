import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.2)
model.save('mnist_digit_recognition_model.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

eval_results = f"Test loss: {score[0]}\nTest accuracy: {score[1]}"
with open("../evals/numbers/keras_numbers_eval.txt", "w") as f:
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
print("Saved evaluation results to emnist_crnn_eval.txt.")


plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Model Training History')
plt.legend()
plt.grid(True)
plt.show()
