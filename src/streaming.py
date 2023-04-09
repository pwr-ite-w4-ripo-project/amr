import cv2 as cv
import numpy as np

def test():
    vid = cv.VideoCapture("abc.mp4")
    if (vid.isOpened() == False): 
        print("Error opening video stream or file")
    
    while(vid.isOpened()):
        ret, frame = vid.read()

        kernel = [
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1],
        ]
        kernel = np.asarray(kernel)
        filtered_frame = cv.filter2D(frame, -1, kernel=kernel)

        if ret == True:
            cv.imshow('Frame', filtered_frame)
            
            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
          
        # Break the loop
        else: 
          break
      
    vid.release()
    cv.destroyAllWindows()