import os
import cv2
import numpy as np

def increase_brightness(img, value=55):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def set_brightness(img,value=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v[v] = value
    
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def illumination(inp_dest, out_dest, eval_frames):
    files = os.listdir(inp_dest) #"illumination/input"
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
    

    
    for f in range(1,l):
        img = cv2.imread(os.path.join(inp_dest, files[f])) #"illumination/input"
        bg = cv2.imread(os.path.join(inp_dest, files[f-1]))
        
        cv2.imshow("img",img)

        #img=cv2.medianBlur(img, 3)
        #bg=cv2.medianBlur(bg, 3)
        cv2.blur(img, (3,3), img)
        cv2.blur(bg, (3,3), bg)

        img=increase_brightness(img)
        bg=increase_brightness(bg)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)

        mask = cv2.absdiff(img, bg)
        
        cv2.blur(mask,(5,5),mask)
        
        mask[mask>17] = 255
        mask[mask<=17] = 0
        
        # mask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel2)
        # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel2)
        # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel2)
        
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel1)
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
        #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
        
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel2)
        #m = cv2.resize(mask, (320,240))
        cv2.imshow("mask", mask)
        
        cv2.waitKey(1)
        if f>=start and f<=end:
            cv2.imwrite(os.path.join(out_dest,"gt"+files[f][2:-3]+"png"),mask)
        
    cv2.destroyAllWindows()
