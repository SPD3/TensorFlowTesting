import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

dataSetPath = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
columnNames = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv(dataSetPath, names=columnNames,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset.tail()

dataset.isna().sum()
dataset = dataset.dropna()

dataset["Origin"] = dataset["Origin"].map({1 : "USA", 2 : "Europe", 3 : "Japan"})
dataset = pd.get_dummies(dataset, prefix="", prefix_sep="")
dataset.tail()

trainDataset = dataset.sample(frac=0.8, random_state=0)
testDataset = dataset.drop(trainDataset.index)

train_stats = trainDataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

trainLabels = trainDataset.pop("MPG")
testLabels = testDataset.pop("MPG")

def norm(x):
    return (x - train_stats["mean"]) / train_stats["std"]

normedTrainData = norm(trainDataset)
normedTestData = norm(testDataset)

def buildModel():
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=[len(trainDataset.keys())]),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])

    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mae", "mse"]
    )
    return model

model = buildModel()

model.summary()

earlyStop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

exampleBatch = normedTestData[:10]
exampleResult = model.predict(exampleBatch)
print(exampleResult)

EPOCHS = 50

history = model.fit(
    normedTrainData,
    trainLabels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[earlyStop, tfdocs.modeling.EpochDots()]
)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.tail()

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

loss, mae, mse = model.evaluate(normedTestData, testLabels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

testPredictions = model.predict(normedTestData).flatten()

a = plt.axes(aspect="equal")
plt.scatter(testLabels, testPredictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0,50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = testPredictions - testLabels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

plt.show()