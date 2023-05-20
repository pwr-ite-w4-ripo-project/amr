# from src.streaming import from_file
from src.filters import test_filter
# from models.test_model import train_model
import cv2 as cv
import tensorflow_hub as hub
import tensorflow as tf

image1 = cv.cvtColor(cv.imread("dataset/images/00012112.png"), cv.COLOR_BGR2RGB) 
image1 = cv.resize(image1, (224, 224))

detector = hub.load("models/detector")
labels = ["person", "box"]

def main():
    # from_file("abc.mp4", test_filter, 0.025)
    # train_model()
    # print(detector)
    # rgb = image1 #cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # rgb_tensor = tf.convert_to_tensor(image1, dtype=tf.float32, name="inputs")
    # rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    # Creating prediction
    # xx = detector.
    rgb_tensor = tf.convert_to_tensor(image1, dtype=tf.float32)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)



    # result = test_filter(image1)
    boxes, scores = detector(rgb_tensor)
    print(boxes[0])
    print(scores)


    return
    while True:
      cv.imshow("xd", image1)
      if cv.waitKey(25) & 0xFF == ord('q'):
          break
    # print(xx)
    
    # testowanie zdj:
    # pred_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # predictions = pred_model.predict(image2[None, :])
    # print(predictions[0])
    
    

if __name__ == "__main__":
    main()