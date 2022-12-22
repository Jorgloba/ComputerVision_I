import numpy as np
import matplotlib.pyplot as plt
import cv2 
from sklearn.cluster import MeanShift, estimate_bandwidth

def plotHisto(image,binSize=1):
    
    # Flatten image
    imageFlat = np.array(image)
    imageFlat = imageFlat.flatten()

    # Get the number of bins
    numberOfBins = round(256/binSize)

    # Plot histogram
    plt.hist(imageFlat,numberOfBins,range=(0,256))
    plt.show() 

def kMeansClustering(image,K,maxIter):

    Z = image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, maxIter, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    return res2

    # Adapted from: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html

def mShiftClustering(image,num): #CORREGIR

    Z = image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    bandwidth = estimate_bandwidth(Z,quantile=0.1,n_samples=100)
    msCluster = MeanShift(bandwidth=bandwidth,bin_seeding=True)
    # Now convert back into uint8, and make original image

    return res2
