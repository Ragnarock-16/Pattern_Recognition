import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from tqdm import tqdm 


def get_image(img_path):
    img = cv2.imread(img_path)
    return img

class FuzzyCMeans:

    def __init__(self, data, epsilon, k=2, m=1.25):
        self.epsilon = epsilon
        self.k = k if k >= 2 else 2
        self.m = m if m >= 1.25 else 1.25
        self.data = data.reshape(-1, 3)  
        self.n, self.d = self.data.shape  

    def initialize_U(self):
        weight = np.random.dirichlet(np.ones(self.k), size=self.n)
        return weight

    def compute_centroid(self):
        weights = np.power(self.u, self.m)  
        numerator = np.dot(weights.T, self.data)  
        denominator = np.sum(weights, axis=0)[:, np.newaxis] 

        self.centroid = numerator / denominator
    
    def update_U(self):
        distances = np.linalg.norm(self.data[:, np.newaxis] - self.centroid, axis=2)

        distances = np.fmax(distances, np.finfo(np.float64).eps)

        new_u = np.zeros_like(self.u)

        for i in range(self.n):
            for j in range(self.k):
                denom = 0
                for c in range(self.k):
                    denom += (distances[i, j] / distances[i, c]) ** (2 / (self.m - 1))
                
                new_u[i, j] = 1 / denom

        return new_u

    def train(self, max_iter=10):

        self.u = self.initialize_U()  
        
        with tqdm(total=max_iter, desc='Training', unit='iteration') as pbar:

            for iteration in range(max_iter):
                self.compute_centroid()  
                u_k = self.update_U()  
                norm = np.linalg.norm(u_k - self.u)
                print('norm', norm)

                if norm < self.epsilon:
                    print(f"\nConverged at iteration {iteration+1}")
                    break
                
                self.u = u_k
                pbar.update(1)  

        return self.centroid, self.u, iteration+1

def visualize_fuzzy_clusters(img, u, eps, k, m, conver):
    num_clusters = u.shape[1]
    fig, axes = plt.subplots(1, num_clusters + 1, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    for j in range(num_clusters):
            membership_map = u[:, j].reshape(img.shape[0], img.shape[1])
            
            heatmap = axes[j + 1].imshow(membership_map, cmap='viridis')  
            axes[j + 1].set_title(f'Membership to Cluster {j + 1}')
            axes[j + 1].axis('off')
            plt.colorbar(heatmap, ax=axes[j + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'/Users/nour/Documents/M2/rf/tp 1/plot/K{k}_M{m}_E{eps}_C{conver}.png')


if __name__ == "__main__":
    img = get_image('/Users/nour/Documents/M2/rf/tp 1/creation_pillar.png')
    img = img / 255.0  
    eps = 0.3
    k = 4
    m = 2
    f_c_means_1 = FuzzyCMeans(img, eps, k, m)
    centroid, u, conver= f_c_means_1.train(max_iter=100)

    visualize_fuzzy_clusters(img, u, eps, k ,m, conver)  