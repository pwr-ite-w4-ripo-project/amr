import numpy as np
import cv2 as cv
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

# kernel = [
#     [-1, -1, -1],
#     [-1,  8, -1],
#     [-1, -1, -1],
# ]

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

def test_filter(image):
    # return cv.filter2D(image, -1, kernel=np.asarray(kernel))
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    # Creating prediction
    boxes, scores, classes, num_detections = detector(rgb_tensor)

    # Processing outputs
    pred_labels = classes.numpy().astype('int')[0] 
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    # Putting the boxes and labels on the image
    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if label != "person":
            continue

        if score < 0.5:
            continue

        score_txt = f'{100 * round(score)}%'
        img_boxes = cv.rectangle(rgb, (xmin, ymax),(xmax, ymin),(0,255,0),2)      
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_boxes, label,(xmin, ymax-10), font, 1.5, (255,0,0), 2, cv.LINE_AA)
        cv.putText(img_boxes,score_txt,(xmax, ymax-10), font, 1.5, (255,0,0), 2, cv.LINE_AA)
        # cv.addWeighted(image, 0.5, img_boxes, 0.5, 0.5)
        # add on frame 
        return img_boxes
