from src.streaming import from_file
from src.filters import test_filter
from models.test_model import train_model

def main():
    # from_file("abc.mp4", test_filter, 0.025)
    train_model()
    
    

if __name__ == "__main__":
    main()