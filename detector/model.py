import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split

classes = ["box", "person"]

# model creation
inputs = tf.keras.Input(shape=(224, 224, 3))
flatten = tf.keras.layers.Flatten()(inputs)

## bounding boxs input
bboxHead = tf.keras.layers.Dense(128, activation="relu")(flatten)
bboxHead = tf.keras.layers.Dense(64, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(32, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

## class labels input
softmaxHead = tf.keras.layers.Dense(512, activation="relu")(flatten)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(512, activation="relu")(softmaxHead)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(len(classes), activation="softmax", name="class_label")(softmaxHead)

model = tf.keras.Model(
    inputs=inputs,
    outputs=[bboxHead, softmaxHead]
)

# training dataset preparation
data = []
labels = []
bboxes = []
paths = []

lines = open("dataset/labels.txt")
for line in lines:
    (filename, label, x, y, width, height) = line.split(" ")

    path = f"dataset/images/{filename}"
    image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    image_as_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    data.append(np.array(image, dtype="float32") / 255.0)
    labels.append(label)
    bboxes.append((x, y, width, height))
    paths.append(path)

data = np.array(data)
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
paths = np.array(paths)

labelBinarizer = LabelBinarizer()
labels = labelBinarizer.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)

split = train_test_split(data, labels, bboxes, paths, test_size=0.20, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

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

trainTargets = {
	"class_label": trainLabels,
	"bounding_box": trainBBoxes
}
testTargets = {
	"class_label": testLabels,
	"bounding_box": testBBoxes
}

model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=32,
	epochs=20,
	verbose=1
)
model.save("models/detector")
