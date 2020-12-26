import functools

import numpy as np
import tensorflow as tf
import pandas as pd

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

trainFilePath = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
testFilePath = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)

labelColumn = "survived"
LABELS = [0, 1]


def getDataSet(filePath, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        filePath,
        batch_size=5,
        label_name=labelColumn,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs
    )
    return dataset


rawTrainData = getDataSet(trainFilePath)
rawTestData = getDataSet(testFilePath)


def showBatch(dataset):
    for batch, label, in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


CSVCOLUMNS = ["survived", "sex", "age", "n_siblings_spouses", "parch", "fare",
              "class", "deck", "embark_town", "alone"]

tempDataset = getDataSet(trainFilePath, column_names=CSVCOLUMNS)

SELECTCOLUMNS = ["survived", "age", "n_siblings_spouses", "parch", "fare"]
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]


def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label


def trySelectColumns():
    tempDataset = getDataSet(
        trainFilePath,
        select_columns=SELECTCOLUMNS,
        column_defaults=DEFAULTS
    )

    showBatch(tempDataset)

    packedDataset = tempDataset.map(pack)
    print()
    print()
    print("------------------------")
    for features, labels in packedDataset.take(1):
        print(features.numpy())
        print()
        print(labels.numpy())


class PackNumericFeatures(object):
    def __init__(self, names):
        super().__init__()
        self.names = names

    def __call__(self, features, labels):
        numericFeatures = [features.pop(name) for name in self.names]
        numericFeatures = [tf.cast(feat, tf.float32)
                           for feat in numericFeatures]
        numericFeatures = tf.stack(numericFeatures, axis=-1)
        features["numeric"] = numericFeatures
        return features, labels


NUMERICFEATURES = ["age", "n_siblings_spouses", "parch", "fare"]

packedTrainData = rawTrainData.map(PackNumericFeatures(NUMERICFEATURES))
packedTestData = rawTestData.map(PackNumericFeatures(NUMERICFEATURES))

print("--------------------")
showBatch(packedTrainData)

exampleBatch, labelsBatch = next(iter(packedTrainData))

desc = pd.read_csv(trainFilePath)[NUMERICFEATURES].describe()
print(desc)

MEAN = np.array(desc.T["mean"])
STD = np.array(desc.T["std"])


def normalizeNumericData(data, mean, std):
    return (data - mean) / std


normalizer = functools.partial(normalizeNumericData, mean=MEAN, std=STD)
numericColumn = tf.feature_column.numeric_column("numeric",
                                                 normalizer_fn=normalizer,
                                                 shape=[len(NUMERICFEATURES)])
numericColumns = [numericColumn]

print(exampleBatch["numeric"])

numericLayer = tf.keras.layers.DenseFeatures(numericColumns)
print()
print()
print(numericLayer(exampleBatch).numpy())
print()
print()

CATEGORIES = {
    'sex': ['male', 'female'],
    'class': ['First', 'Second', 'Third'],
    'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone': ['y', 'n']
}

categoricalColumns = []
for feature, vocab in CATEGORIES.items():
    catCol = tf.feature_column.\
        categorical_column_with_vocabulary_list(key=feature,
                                                vocabulary_list=vocab)
    categoricalColumns.append(tf.feature_column.indicator_column(catCol))

categoricalLayer = tf.keras.layers.DenseFeatures(categoricalColumns)

print()
print()
print()
print(categoricalLayer(exampleBatch).numpy()[0])
print(categoricalColumns)
print(numericColumns)
print()
print()
print()

preprocessingLayer = [tf.keras.layers.
                      DenseFeatures(categoricalColumns+numericColumns)]

model = tf.keras.Sequential([
    preprocessingLayer,
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"]
)

trainData = packedTrainData.shuffle(500)
testData = packedTestData

model.fit(trainData, epochs=100)

test_loss, test_accuracy = model.evaluate(testData)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(testData)

for prediction, survived in zip(predictions[:10], list(testData)[0][1][:10]):
    prediction = tf.sigmoid(prediction).numpy()
    print("Predicted survival: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("SURVIVED" if bool(survived) else "DIED"))
