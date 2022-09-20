#### Lab03_Color_AlphaBeta_Adjustedment.py ####
import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt
# Global Variables
beta_brightness_value = 100
alpha_contrast_value = 10
source_img = np.zeros((10,10,3), dtype=np.uint8)
adjusted_img = np.zeros((10,10,3), dtype=np.uint8)
hist_img = np.zeros((10,10,3), dtype=np.uint8)
 
def handler_adjustAlphaBeta(x):
    global beta_brightness_value,alpha_contrast_value
    global source_img,adjusted_img,hist_img
    beta_brightness_value = cv.getTrackbarPos('beta','BrightnessContrast')
    alpha_contrast_value = cv.getTrackbarPos('alpha','BrightnessContrast')
    alpha = alpha_contrast_value / 10
    beta = int(beta_brightness_value - 100)
    print(f"alpha={alpha} / beta={beta}")
    
    ## loop access each pixel -> too slow
    ''' for y in range(source_img.shape[0]):
        for x in range(source_img.shape[1]):
            for c in range(source_img.shape[2]):
                adjusted_img[y,x,c] = np.clip( alpha * source_img[y,x,c] + beta , 0, 255)
    '''
    # for better performance, pls use -> dst = cv.addWeighted(src1, alpha, beta, 0.0, src2)
    adjusted_img = cv.convertScaleAbs(source_img, alpha=alpha, beta=beta)
 
    # Update histogram
    bgr_planes = cv.split(adjusted_img)
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX) # normalize -> 0 - 1
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX) # normalize -> 0 - 1
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX) # normalize -> 0 - 1
    for i in range(1, histSize):
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(b_hist[i]) ),
                ( 255, 0, 0), thickness=2)
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(g_hist[i]) ),
                ( 0, 255, 0), thickness=2)
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(r_hist[i]) ),
                ( 0, 0, 255), thickness=2)
 
def main():
    global beta_brightness_value,alpha_contrast_value
    global source_img,adjusted_img,hist_img
 
    if(len(sys.argv)>=2):
        source_img = cv.imread(str(sys.argv[1]))
    else : 
        source_img = cv.imread("output.jpg", 1)
 
    #source_img = cv.cvtColor(source_img,cv.COLOR_BGR2GRAY)
 
    #named windows
    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    cv.namedWindow("BrightnessContrast", cv.WINDOW_NORMAL)
    cv.namedWindow("Histogram", cv.WINDOW_NORMAL)
 
    #create trackbar
    cv.createTrackbar('beta', 'BrightnessContrast', beta_brightness_value, 200, handler_adjustAlphaBeta)
    cv.createTrackbar('alpha', 'BrightnessContrast', alpha_contrast_value, 50, handler_adjustAlphaBeta)

    adjusted_img = source_img.copy()
 
    while(True):
        cv.imshow("Original",source_img)
        cv.imshow("BrightnessContrast",adjusted_img)
        cv.imshow("Histogram",hist_img)
        key = cv.waitKey(100) 
        if(key==27): #ESC = Exit Program
            break
 
    cv.destroyAllWindows()
 
if __name__ == "__main__":
    main()