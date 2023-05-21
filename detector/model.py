import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import train_test_split
import os

# classes = ["person", "box"]
classes = ["person"]

# model creation
#create the base layers
inputs = tf.keras.Input(shape=(216, 216, 3))
base_layers = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)
base_layers = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(base_layers)
base_layers = tf.keras.layers.MaxPooling2D()(base_layers)
base_layers = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(base_layers)
base_layers = tf.keras.layers.MaxPooling2D()(base_layers)
base_layers = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(base_layers)
base_layers = tf.keras.layers.MaxPooling2D()(base_layers)
flatten = tf.keras.layers.Flatten()(base_layers)

## bounding boxs input
bboxHead = tf.keras.layers.Dense(128, activation="relu")(flatten)
# bboxHead = tf.keras.layers.Dense(64, activation="relu")(bboxHead)
# bboxHead = tf.keras.layers.Dense(32, activation="relu")(bboxHead)
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

# files = os.listdir("dataset/labels")
# for file in files:
#     filename = file.replace("txt", "png")
#     lines = open(f"dataset/labels/{file}")

#     path = f"dataset/images/{filename}"
#     image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
#     image_as_arr = tf.keras.preprocessing.image.img_to_array(image)
#     data.append(np.array(image, dtype="float32") / 255.0)
#     paths.append(path)

#     image_bboxes = []
#     image_labels = []
#     for line in lines:
#         if (len(line) < 2):
#             continue

#         (label, x, y, width, height) = line.strip().split(" ")
        
#         image_labels.append(int(label))
#         bboxes.append(np.array([float(x), float(y), float(x) + float(width), float(y) + float(height)], dtype="float32"))

#     # bboxes.append(image_bboxes)
#     labels.append(image_labels)

lines = open("dataset/labels.txt")
for line in lines:
    (filename, label, x, y, width, height) = line.strip().split(" ")

    if (label == 1 or label == "1"):
        continue

    path = f"dataset/images/{filename}"
    image = tf.keras.preprocessing.image.load_img(path, target_size=(216, 216))
    image_as_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    data.append(np.array(image, dtype="float32") / 255.0)
    labels.append(int(label))
    bboxes.append((
        round(float(x), 2), 
        round(float(y), 2), 
        round(float(x) + float(width), 2), 
        round(float(y) + float(height), 2)
    ))
    # print(bboxes[len(bboxes) - 1])
    paths.append(path)

data = np.array(data)
labels = np.array(labels, dtype="int32")
bboxes = np.array(bboxes, dtype="float32")
paths = np.array(paths)

print(bboxes)

# labelBinarizer = MultiLabelBinarizer()
labelBinarizer = LabelBinarizer()
labels = labelBinarizer.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)

print(len(data))
print(len(labels))
print(len(bboxes))
print(len(paths))

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
model.save("models/detector7")
