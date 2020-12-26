
import numpy as np
import tensorflow as tf

DATA_URL = """https://storage.googleapis.com/tensorflow/
              tf-keras-datasets/mnist.npz"""
path = tf.keras.utils.get_file("mnist.npz", DATA_URL)

with np.load(path) as data:
    trainExamples = data["x_train"]
    trainLabels = data["y_train"]
    testExamples = data["x_test"]
    testLabels = data["y_test"]

trainDataset = tf.data.Dataset.from_tensor_slices((trainExamples, trainLabels))
testDataset = tf.data.Dataset.from_tensor_slices((testExamples, testLabels))

BATCHSIZE = 64
SHUFFLEBATCHSIZE = 100

trainDataset = trainDataset.shuffle(SHUFFLEBATCHSIZE).batch(BATCHSIZE)
testDataset = testDataset.batch(BATCHSIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10)
])


model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["sparse_categorical_accuracy"]
)

model.fit(trainDataset, epochs=3)

model.evaluate(testDataset)
