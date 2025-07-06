# main.py
from clustering import load_data, preprocess_data, apply_kmeans
from visualize import plot_elbow, plot_clusters
# main.py
from gui import run_gui



def main():
    df = load_data("dataset/Mall_Customers.csv")
    X = preprocess_data(df)
    plot_elbow(X)
    kmeans_model, labels = apply_kmeans(X, n_clusters=5)
    plot_clusters(X, labels, kmeans_model)


if __name__ == "__main__":
    main()
    run_gui()