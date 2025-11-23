import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA

if __name__ == "__main__":
    image_raw = imread("photo.jpg")
    image_sum = image_raw.sum(axis=2)
    image_bw = image_sum / image_sum.max()
    
    print(image_raw.shape)
    print(image_sum.shape)
    print(image_bw.max())
    plt.imshow(image_bw, cmap='gray')
    plt.axis('off')
    plt.show()
    
    

    