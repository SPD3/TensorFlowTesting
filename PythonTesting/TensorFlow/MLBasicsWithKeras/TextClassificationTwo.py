import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

(trainData, testData), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True
)

encoder = info.features["text"].encoder
print ('Vocabulary size: {}'.format(encoder.vocab_size))

for trainExample, trainLabel in trainData.take(1):
    print("Encoded text: ", trainExample[:10].numpy())
    print("Label: ", trainLabel.numpy())

BUFFER_SIZE = 1000
trainBatches =  (
    trainData
    .shuffle(BUFFER_SIZE)
    .padded_batch(32)
)

testBatches = (
    testData
    .padded_batch(32)
)

model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size,16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1)
])

model.summary()

model.compile(
    optimizer="adam",
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(trainBatches, epochs=10, validation_data=testBatches, validation_steps=30)

loss, accuracy = model.evaluate(testBatches)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

historyDict = history.history
historyDict.keys()

acc = historyDict['accuracy']
val_acc = historyDict['val_accuracy']
loss = historyDict['loss']
val_loss = historyDict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()