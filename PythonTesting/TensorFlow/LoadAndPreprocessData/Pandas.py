import pandas as pd
import tensorflow as tf

link = 'https://storage.googleapis.com/applied-dl/heart.csv'
csv_file = tf.keras.utils.get_file('heart.csv', link)

df = pd.read_csv(csv_file)

df["thal"] = pd.Categorical(df["thal"])
df["thal"] = df.thal.cat.codes

target = df.pop("target")
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feat, targ in dataset.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))
tf.constant(df["thal"])

trainDataset = dataset.shuffle(len(df)).batch(1)


def getCompiledModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model


model = getCompiledModel()

# model.fit(trainDataset, epochs=15)

inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)
x = tf.keras.layers.Dense(10, activation="relu")(x)

output = tf.keras.layers.Dense(1)(x)

modelFunc = tf.keras.Model(inputs=inputs, outputs=output)

modelFunc.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

dictSlices = tf.data.Dataset.from_tensor_slices((df.to_dict("list"),
                                                 target.values)).batch(16)

modelFunc.fit(dictSlices, epochs=15)