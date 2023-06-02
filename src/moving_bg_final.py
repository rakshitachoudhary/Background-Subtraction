import os
import cv2
import numpy as np
#0.4617
def moving_bg(inp_dest, out_dest, eval_frames):
    files = os.listdir(inp_dest) #"moving_bg/input"
    files.sort()
    l = len(files)

    f = open(eval_frames, 'r')
    line = f.readline()
    words = line.split()
    start = int(words[0])-1
    end = int(words[1])-1

    kernel1=np.ones((7,7),np.uint8)
    kernel2=np.ones((7,7),np.uint8)
    #kernel1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    #kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

    # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    # fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
    fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    for f in range(1,l):
        img = cv2.imread(os.path.join(inp_dest, files[f]),0)
        cv2.imshow("img",img)
        # cv2.blur(img, (10,10), img)
        img=cv2.medianBlur(img, 5)
        img=cv2.medianBlur(img, 3)
        
        mask = fgbg.apply(img)

        #mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel2)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)

        cv2.imshow("mask", mask)
        cv2.waitKey(1)
        if f>=start and f<=end:
            cv2.imwrite(os.path.join(out_dest,"gt"+files[f][2:-3]+"png"),mask)
        
    cv2.destroyAllWindows()