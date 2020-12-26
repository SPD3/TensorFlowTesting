import os
import tensorflow as tf
from tensorflow import keras
print(tf.version.VERSION)

(trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.mnist.load_data()

trainLabels = trainLabels[:1000]
testLabels = testLabels[:1000]

trainImages = trainImages[:1000].reshape(-1, 28 * 28) / 255.0
testImages = testImages[:1000].reshape(-1, 28 * 28) / 255.0

def createModel():
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation="relu", input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model

checkpointPath = "training_2/cp-{epoch:04d}.ckpt"
checkpointDir = os.path.dirname(checkpointPath)

cpCallback = keras.callbacks.ModelCheckpoint(
    filepath=checkpointPath,
    save_weights_only=True,
    verbose=1,
    save_freq=5
)

model = createModel()
model.save_weights(checkpointPath.format(epoch=0))

def fitModel():
    model.fit(
        trainImages, 
        trainLabels, 
        epochs=50,
        callbacks=[cpCallback],
        validation_data=(testImages, testLabels),
        verbose=0
    )

#fitModel()

def testLoading():
    model = createModel()
    loss, acc = model.evaluate(testImages, testLabels, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

    latest = tf.train.latest_checkpoint(checkpointDir)
    model.load_weights(latest)

    loss, acc = model.evaluate(testImages, testLabels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#testLoading()

def manualSavingtest(model):
    model.save_weights("./checkpoints/my_checkpoint")
    model = createModel()   
    model.load_weights("./checkpoints/my_checkpoint")
    loss, acc = model.evaluate(testImages, testLabels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#manualSavingtest(model)

def saveEntireModelTest():
    model = createModel()
    model.fit(trainImages, trainLabels, epochs=5)
    model.save("saved_model/my_model")

#saveEntireModelTest()

def loadEntireModel():
    newModel = keras.models.load_model("saved_model/my_model")
    newModel.summary()
    loss, acc = newModel.evaluate(testImages,  testLabels, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

    print(newModel.predict(testImages).shape)

#loadEntireModel()

def saveEntireModelh5Format():
    model = createModel()
    model.fit(trainImages, trainLabels, epochs=5)
    model.save("saved_model/my_model.h5")

    newModel = keras.models.load_model("saved_model/my_model.h5")
    loss, acc = newModel.evaluate(testImages,  testLabels, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

saveEntireModelh5Format()