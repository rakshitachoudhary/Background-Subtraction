import os
import cv2
import numpy as np

def baseline(inp_dest, out_dest, eval_frames):
    files = os.listdir(inp_dest)   #"baseline/input"
    files.sort()
    l = len(files)

    f = open(eval_frames, 'r')
    line = f.readline()
    words = line.split()
    start = int(words[0])-1
    end = int(words[1])-1
    #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel1=np.ones((5,5),np.uint8)
    kernel2=np.ones((5,5),np.uint8)


    fgbg=cv2.createBackgroundSubtractorKNN(detectShadows=False)

    for f in range(0,l):
        frame = cv2.imread(os.path.join(inp_dest, files[f]),0)
        cv2.imshow("Frame",frame)
        ede=fgbg.apply(frame)
        fgmask = fgbg.apply(frame)
        
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
        fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel1)
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
        
        fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel2)
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
        
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
        
        cv2.imshow("FG Mask",fgmask)

        if f>=start and f<=end:
            cv2.imwrite(os.path.join(out_dest,"gt"+files[f][2:-3]+"png"),fgmask)
        #"baseline/results"
        cv2.waitKey(1)
        
    cv2.destroyAllWindows()
