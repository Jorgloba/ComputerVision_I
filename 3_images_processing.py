import numpy as np
import cv2 
import matplotlib.pyplot as plt


intro_images_path='./Images/Intro_images/'
output_path='./Images/Outputs/'
template_path = "./Images/Intro_images/Tuna_template.jpg"
image_path = "./Images/Intro_images/Tuna_Relative_Sizes.jpg"
print('Image paths ....')




img = cv2.imread(intro_images_path+'Tuna_Relative_Sizes.jpg',0)
width_img, height_img = img.shape[::-1]

template = cv2.imread(intro_images_path+'Tuna_template.jpg',0)
width_temp, height_temp = template.shape[::-1]

# All the 6 methods for comparison in a list
#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

for method in methods:
    print('Running comparison method ' + method + '...')
    
    while height_temp <= height_img and width_temp <= width_img:
        res = cv2.matchTemplate(img,template, eval(method))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        threshold=0.6
        loc = np.where(res >= threshold)
        
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
            bottom_right = (top_left[0] + width_temp, top_left[1] + height_temp)
        
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + width_temp, pt[1] + height_temp), [0,255,0], 1)
        
        template = cv2.pyrUp(template)
        width_temp, height_temp  = template.shape[::-1]
    

    plt.imshow(img, cmap = 'gray')
    plt.title('Detection area with: ' + method), plt.xticks([]), plt.yticks([])
    plt.show()
