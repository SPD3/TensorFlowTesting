import tensorflow as tf
from tensorflow import keras

import IPython
import kerastuner as kt

(trainImages, trainLabels), (testImages, testLabels) = keras.datasets.fashion_mnist.load_data()

trainImages = trainImages.astype("float32") / 255.0
testImages = testImages.astype("float32") / 255.0

def modelBuilder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    
    hpUnits = hp.Int("units", min_value=32, max_value=512, step=32)
    
    model.add(keras.layers.Dense(units=hpUnits, activation= "relu"))
    model.add(keras.layers.Dense(10))

    hpLearningRate = hp.Choice("learning_rate", values = [1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hpLearningRate),
        loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model

tuner = kt.Hyperband(
    modelBuilder,
    objective = "val_accuracy",
    max_epochs = 10,
    factor = 3,
    directory = "my_dir",
    project_name = 'intro_to_kt'
)

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        IPython.display.clear_output(wait = True)
        return super().on_train_end(logs=logs)

tuner.search(trainImages, trainLabels, epochs = 10, validation_data = (testImages, testLabels), callbacks = [ClearTrainingOutput()])

bestHps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {bestHps.get('units')} and the optimal learning rate for the optimizer
is {bestHps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(bestHps)
model.fit(trainImages, trainLabels, epochs=10, validation_data = (testImages, testLabels))