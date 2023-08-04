import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from emnist import list_datasets, extract_training_samples, extract_test_samples


x_train, y_train = extract_training_samples('byclass')
x_test, y_test = extract_test_samples('byclass')
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape = (28, 28, 1)
inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Reshape(target_shape=(5, 5 * 64))(x)
x = layers.GRU(128, return_sequences=True)(x)
x = layers.GRU(128, return_sequences=False)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
batch_size = 256
epochs = 125
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save("emnist_crnn.h5")
print("Saved model to disk.")
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


eval_results = f"Test loss: {score[0]}\nTest accuracy: {score[1]}"
with open("evals/emnist_crnn_eval.txt", "w") as f:
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

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training Metrics')
plt.legend()
plt.show()
