# visualize.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_elbow(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid()
    plt.show()

def plot_clusters(X, labels, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[labels == 0, 0], X[labels == 0, 1], c='red', label='Cluster 1')
    plt.scatter(X[labels == 1, 0], X[labels == 1, 1], c='blue', label='Cluster 2')
    plt.scatter(X[labels == 2, 0], X[labels == 2, 1], c='green', label='Cluster 3')
    plt.scatter(X[labels == 3, 0], X[labels == 3, 1], c='cyan', label='Cluster 4')
    plt.scatter(X[labels == 4, 0], X[labels == 4, 1], c='magenta', label='Cluster 5')
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title("Customer Segments")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.show()
