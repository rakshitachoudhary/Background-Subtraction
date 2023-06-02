import os
import cv2
import numpy as np

def ptz(inp_dest, out_dest, eval_frames):
    files = os.listdir(inp_dest) 
    files.sort()
    l = len(files)

    f = open(eval_frames, 'r')
    line = f.readline()
    words = line.split()
    start = int(words[0])-1
    end = int(words[1])-1

    frame = cv2.imread(os.path.join(inp_dest, files[0]))
    hsv_image = np.zeros_like(frame)
    hsv_image[...,1] = 255

    for f in range(1,l):
        Image = cv2.imread(os.path.join(inp_dest, files[f]),0)
        Prev = cv2.imread(os.path.join(inp_dest, files[f-1]),0)
        
        cv2.imshow("Image",Image)
        
        flow = cv2.calcOpticalFlowFarneback(Prev,Image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flowp = cv2.split(flow)
        magnitude, angles = cv2.cartToPolar(flowp[0], flowp[1])
        hsv_image[...,0] = angles*180/np.pi/2
        hsv_image[...,2] = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
        mask = cv2.cvtColor(hsv_image,cv2.COLOR_HSV2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask[mask>50] = 255
        mask[mask<=50] = 0
        cv2.imshow('Mask',mask)
        cv2.waitKey(1)
        if f>=start and f<=end:
            cv2.imwrite(os.path.join(out_dest,"gt"+files[f][2:-3]+"png"),mask)

    cv2.destroyAllWindows()