import cv2 as cv
import time

def from_file(path, filter, delayBetweenFrames):
    vid = cv.VideoCapture(path)
    if (vid.isOpened() == False): 
        print("Error opening video stream or file")
    
    while(vid.isOpened()):
        ret, frame = vid.read()
        filtered_frame = filter(frame)

        # time.sleep(delayBetweenFrames)
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