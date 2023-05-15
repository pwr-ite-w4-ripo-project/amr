import tensorflow as tf
import tensorflow_datasets as tfds

classes = ["box", "person", "unknown"]

dataset = tf.keras.utils.image_dataset_from_directory(
    directory="better_dataset/training",
    image_size=(128, 128),
    subset="training",
    validation_split=0.2,
    seed=123
)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(len(classes), activation=tf.nn.softmax)
])

def train_model():
    print(dataset.class_names)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        x=dataset,
        epochs=10
    )

    model.save("models/dummy")
