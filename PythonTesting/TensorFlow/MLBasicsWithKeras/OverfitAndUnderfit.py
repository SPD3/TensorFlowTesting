import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

print(tf.__version__)

from IPython import display
from matplotlib import pyplot as plt 
import numpy as np

import pathlib
import shutil
import tempfile

logDir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logDir, ignore_errors=True)

gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

FEATURES = 28
ds = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES+1), compression_type="GZIP")

def packRow(*row):
    labels = row[0]
    features = tf.stack(row[1:], 1)
    return features, labels

packedDs = ds.batch(10000).map(packRow).unbatch()

for feature, label in packedDs.batch(1000).take(1):
    print (feature[0])
    

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validateDs = packedDs.take(N_VALIDATION).cache()
trainDs = packedDs.skip(N_VALIDATION).take(N_TRAIN).cache()

validateDs = validateDs.batch(BATCH_SIZE)
trainDs = trainDs.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

lrSchedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*1000,
    decay_rate=1,
    staircase=False
)

def getOptimizer():
    return tf.keras.optimizers.Adam(lrSchedule)

step = np.linspace(0, 1e5)
lr = lrSchedule(step)

def getCallbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor="val_binary_crossentropy", patience=200),
        tf.keras.callbacks.TensorBoard(logDir/name)
    ]

def compileAndFit(model, name, optimizer=None, maxEpochs=10000):
    if optimizer == None:
        optimizer = getOptimizer()
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                name="binary_crossentropy"
            ),
            ["accuracy"]
        ]
    )

    model.summary()

    history = model.fit(
        trainDs,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=maxEpochs,
        validation_data=validateDs,
        callbacks=getCallbacks(name),
        verbose=0
    )

    return history

tinyModel = tf.keras.Sequential([
    layers.Dense(16, activation="elu", input_shape=(FEATURES,)),
    layers.Dense(1)
])

smallModel = tf.keras.Sequential([
    layers.Dense(16, activation="elu", input_shape=(FEATURES,)),
    layers.Dense(16),
    layers.Dense(1)
])

mediumModel = tf.keras.Sequential([
    layers.Dense(64, activation="elu", input_shape=(FEATURES,)),
    layers.Dense(64, activation="elu"),
    layers.Dense(64, activation="elu"),
    layers.Dense(1),

])

largeModel = tf.keras.Sequential([
    layers.Dense(512, activation="elu", input_shape=(FEATURES,)),
    layers.Dense(512, activation="elu"),
    layers.Dense(512, activation="elu"),
    layers.Dense(512, activation="elu"),
    layers.Dense(1),
])

l2Model = tf.keras.Sequential([
    layers.Dense(512, activation="elu", 
        kernel_regularizer=regularizers.l2(0.001),
        input_shape=(FEATURES,)),
    layers.Dense(512, activation="elu", 
        kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation="elu", 
        kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation="elu", 
        kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

dropoutModel = tf.keras.Sequential([
    layers.Dense(512, activation="elu", input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation="elu"),
    layers.Dropout(0.5),
    layers.Dense(512, activation="elu"),
    layers.Dropout(0.5),
    layers.Dense(512, activation="elu"),
    layers.Dropout(0.5),
    layers.Dense(1)
])

combinedModel = tf.keras.Sequential([
    layers.Dense(512, activation="elu", 
        kernel_regularizer=regularizers.l2(0.001),
        input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation="elu", 
        kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(512, activation="elu", 
        kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(512, activation="elu", 
        kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1)
])

sizeHistories = {}
sizeHistories["Tiny"] = compileAndFit(tinyModel, "sizes/Tiny")
#sizeHistories["Small"] = compileAndFit(smallModel, "sizes/Small")
#sizeHistories["Medium"] =   compileAndFit(mediumModel, "sizes/Medium")
#sizeHistories["Large"] =   compileAndFit(largeModel, "sizes/Large")

display.IFrame(
    src="https://tensorboard.dev/experiment/vW7jmmF9TmKmy3rbheMQpw/#scalars&_smoothingWeight=0.97",
    width="100%", height="800px")

shutil.rmtree(logDir/"regularizers/Tiny", ignore_errors=True)
shutil.copytree(logDir/'sizes/Tiny', logDir/'regularizers/Tiny')
regularizerHistories = {}
regularizerHistories["Tiny"] = sizeHistories["Tiny"]
regularizerHistories["l2"] = compileAndFit(l2Model, "regularizers/l2")
regularizerHistories["dropoutModel"] = compileAndFit(dropoutModel, "regularizers/dropoutModel")
regularizerHistories["combinedModel"] = compileAndFit(combinedModel, "regularizers/combinedModel")

plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(regularizerHistories)
#a = plt.xscale("log")
#plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5,0.7])
#plt.xlabel("Epochs [Log Scale]")
plt.show()
