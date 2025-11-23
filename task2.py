import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA

def reconstruct_image(pca, image_matrix, n_components):
    pca_k = PCA(n_components=n_components)
    transformed = pca_k.fit_transform(image_matrix)
    reconstructed = pca_k.inverse_transform(transformed)
    return reconstructed

def analyze_variance(pca):
    cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
    comp_95perc = np.argmax(cumulative_var >= 95) + 1
    
    print(f"Number of components needed to cover 95% variance: {comp_95perc}")

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_var)
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative Explained variance')
    plt.title('Cumulative Explained Variance explained by the components')
    
    plt.axhline(y=95, color='r', linestyle='--')
    plt.axvline(x=comp_95perc, color='k', linestyle='--')
    plt.grid(True)
    plt.show()

    return comp_95perc

def plot_comparisons(image_matrix):
    component_counts = [5, 15, 25, 75, 100, 170]
    
    plt.figure(figsize=(15, 10))
    
    for i, n in enumerate(component_counts):
        rec_img = reconstruct_image(None, image_matrix, n) 
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(rec_img, cmap='gray')
        plt.title(f"Components: {n}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

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
    
    pca = PCA()
    pca.fit(image_bw)
    
    n_comp = analyze_variance(pca)
    
    reconstructed_image = reconstruct_image(pca, image_bw, n_comp)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.axis('off')
    plt.show()
    
    plot_comparisons(image_bw)
    

    