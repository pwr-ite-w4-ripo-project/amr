# from src.streaming import from_file
# from src.filters import test_filter
from models.test_model import train_model
import tensorflow as tf
import cv2 as cv

# model = tf.keras.models.load_model("./models/dummy")

# image1 = cv.cvtColor(cv.imread("better_dataset/training/box/box-3.png"), cv.COLOR_BGR2RGB) 
# image2 = cv.cvtColor(cv.imread("better_dataset/training/unknown/unknown-3.png"), cv.COLOR_BGR2RGB) 

def main():
    # from_file("abc.mp4", test_filter, 0.025)
    train_model()

    # testowanie zdj:
    # pred_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # predictions = pred_model.predict(image2[None, :])
    # print(predictions[0])
    
    

if __name__ == "__main__":
    main()