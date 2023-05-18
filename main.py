# from src.streaming import from_file
# from src.filters import test_filter
# from models.test_model import train_model
import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv

from tensorflow_hub import object_detector

# model = tf.keras.models.load_model("./models/dummy")

# image1 = cv.cvtColor(cv.imread("better_dataset/training/box/box-3.png"), cv.COLOR_BGR2RGB) 
# image2 = cv.cvtColor(cv.imread("better_dataset/training/unknown/unknown-3.png"), cv.COLOR_BGR2RGB) 

# detector = hub.load("./models/dummy")
def main():
    pass
    # from_file("abc.mp4", test_filter, 0.025)
    # train_model()
    # print(detector)
    # rgb = image1 #cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32, name="inputs")
    # rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    # Creating prediction
    # xx = detector.

    # print(xx)
    
    # testowanie zdj:
    # pred_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # predictions = pred_model.predict(image2[None, :])
    # print(predictions[0])
    
    

if __name__ == "__main__":
    main()