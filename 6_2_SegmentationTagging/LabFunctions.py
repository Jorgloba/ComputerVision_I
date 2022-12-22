import numpy as np
import cv2

def histogram(img):
    # Empty array of 256 zeros (grayscale image space is from 0 to 255)
    hist = np.zeros([256], np.int32) 
    # Looping through pixels in image and incrementing count for the intensity 
    for p in img:
        hist[p] += 1
    return hist

def multi_level_thresholding(img, N):
    # img = cv2.cvtColor(cv2.COLOR_RGB2BGR)
    
    inc = 256 / N
    result = np.zeros(img.shape, dtype=np.uint)
    nrows = img.shape[0]
    ncols = img.shape[1]
    
    for x in range(nrows):
        for y in range(ncols):
            for l in range(N):
                lower_bound = l*inc
                upper_bound = (l+1) * inc
                if img[x,y] >= lower_bound and img[x,y] < upper_bound:
                    result[x,y] = lower_bound
    return result
    

def otsu(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print("No color conversion applied")
    
    nrows, ncols = (img.shape[0], img.shape[1])
    hist, bins = np.histogram(img, bins=256, range=(0,255) )
    
    results = []
    # Looping through each bin of the histogram
    for b in range(256):
        prob_class_a = np.sum(hist[:b+1]) / (nrows * ncols)
        prob_class_b = np.sum(hist[b+1:]) / (nrows * ncols)
        
        # Compute the mean of intensities of each class
        mean_class_a = np.mean(bins[:b+1]) if hist[:b+1].sum() > 0 else 0
        mean_class_b = np.mean(bins[b+1:]) if hist[b+1:].sum() > 0 else 0
        # Compute the inter-class variance as the cuadratic difference between means
        icv = prob_class_a * prob_class_b * (mean_class_a - mean_class_b) ** 2
        results.append(icv)
        
    # Minimise the inter-class variance 
    min_variance = np.argmax(results)
    
    # Loop through image and set pixels to one class or another depending of threeshold
    segmented_img = img.copy()
    for r in range(nrows):
        for c in range(ncols):
            if(segmented_img[r,c] < bins[min_variance]) :
                segmented_img[r,c] = 0
            else:
                segmented_img[r,c] = 1
    
    return segmented_img , min_variance, hist