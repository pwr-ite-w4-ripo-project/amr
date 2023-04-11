from src.streaming import from_file
from src.filters import test_filter

def main():
    from_file("abc.mp4", test_filter, 0.025)

if __name__ == "__main__":
    main()