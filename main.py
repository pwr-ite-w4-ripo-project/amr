from src.streaming import from_file
from src.filters import test_filter
# from models.test_model import train_model
import os

def main():
    # from_file("abc.mp4", test_filter, 0.025)
    # train_model()
    path = "./datasets/abc/train_images"

    for subdir in os.listdir(path):
        print(path + "/" + subdir)
        for file in os.listdir(path + "/" + subdir):
            print("{path}/{subdir}/{file}".format(path=path, subdir=subdir, file=file))
    

if __name__ == "__main__":
    main()