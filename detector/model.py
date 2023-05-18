import tensorflow as tf
# import tensorflow_datasets as tfds

classes = ["box", "person", "unknown"]

# dataset = tf.keras.utils.image_dataset_from_directory(
#     directory="better_dataset/training",
#     image_size=(128, 128),
#     subset="training",
#     validation_split=0.2,
#     seed=123
# )
inputs = tf.keras.Input(shape=(128, 128, 3))

flatten = tf.keras.layers.Flatten()(inputs)

# bounding boxs input
bboxHead = tf.keras.layers.Dense(128, activation="relu")(flatten)
bboxHead = tf.keras.layers.Dense(64, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(32, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

# class labels input
softmaxHead = tf.keras.layers.Dense(512, activation="relu")(flatten)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(512, activation="relu")(softmaxHead)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(len(classes), activation="softmax", name="class_label")(softmaxHead)

model = tf.keras.Model(
    inputs=inputs,
    outputs=[bboxHead, softmaxHead]
)

# def train_model():

model.compile(
    optimizer='adam',
    loss={
        "class_label": "categorical_crossentropy",
        "bounding_box": "mean_squared_error",
    },
    loss_weights={
        "class_label": 1.0,
        "bounding_box": 1.0
    },
    metrics=['accuracy']
)

print(model.summary())
    # model.fit(
    #     x=dataset,
    #     epochs=10
    # )

    # model.save("models/dummy")
