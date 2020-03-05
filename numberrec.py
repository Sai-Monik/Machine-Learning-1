import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
#print(train_labels[0])
#print(train_images[0])
#plt.imshow(train_images[0], cmap=plt.cm.binary)
#plt.show()
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels,epochs=3)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc: ",test_acc)import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
#print(train_labels[0])
#print(train_images[0])
#plt.imshow(train_images[0], cmap=plt.cm.binary)
#plt.show()
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels,epochs=3)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc: ",test_acc)