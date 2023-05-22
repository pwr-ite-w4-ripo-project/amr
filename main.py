# from src.streaming import from_file
from src.filters import test_filter
# from models.test_model import train_model
import cv2 as cv
import tensorflow_hub as hub
import tensorflow as tf

image1 = cv.imread("dataset/images/00039143.png")#), cv.COLOR_BGR2RGB) 
# image1 = cv.imread("dataset/images/00033554.png")#), cv.COLOR_BGR2RGB) 
image2 = cv.resize(image1, (216, 216))

detector = hub.load("models/detector")
labels = ["person", "box"]
# (x, y, width, height) = (0.142708, 0.385185, 0.24375, 0.459259)

def main():
    # from_file("abc.mp4", test_filter, 0.025)
    # train_model()
    # print(detector)
    # rgb = image1 #cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # rgb_tensor = tf.convert_to_tensor(image1, dtype=tf.float32, name="inputs")
    # rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    # Creating prediction
    # xx = detector.
    print(image1)
    rgb_tensor = tf.convert_to_tensor(image2, dtype=tf.float32)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)

    boxes, scores = detector(rgb_tensor)
    print(boxes)
    print(scores)

    # image3 = cv.rectangle(image1, pt1=(int((x - width/2) * 1920), int((y - height/2) * 1080)), pt2=(int((x + width/2) * 1920), int((y + height/2) * 1080)),color=(0,255,0), thickness=2) 
    # image3 = cv.rectangle(image1, pt1=(int(boxes[0][0] * 1920), int(boxes[0][1] * 1080)), pt2=(int((boxes[0][0] + boxes[0][2]) * 1920), int((boxes[0][3]) * 1080)),color=(0,255,0), thickness=2) 
    image3 = cv.rectangle(image1, pt1=(int(boxes[0][0] * 1920), int(boxes[0][1] * 1080)), pt2=(int(boxes[0][2] * 1920), int(boxes[0][3] * 1080)),color=(0,255,0), thickness=2) 

    # return
    while True:
      cv.imshow("xd", image3)
      if cv.waitKey(25) & 0xFF == ord('q'):
          break
    # print(xx)
    

if __name__ == "__main__":
    main()