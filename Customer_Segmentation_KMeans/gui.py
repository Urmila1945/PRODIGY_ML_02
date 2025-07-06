# gui.py (fully centered windows)
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import pandas as pd
from clustering import load_data, preprocess_data, apply_kmeans
from visualize import plot_elbow, plot_clusters
from sklearn.metrics import silhouette_score  # Make sure this import is at the top
class KMeansGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Segmentation - KMeans Clustering")
        self.root.iconbitmap("app_icon.ico")

        self.center_window(self.root, 600, 450)
                
                # Show top logo image


        self.is_dark = tk.BooleanVar(value=False)
        self.toggle_theme()

        self.label = tk.Label(root, text="Customer Segmentation using KMeans", font=("Arial", 16))
        self.label.pack(pady=15)

        self.load_button = tk.Button(root, text="Load Dataset", command=self.load_file, width=25)
        self.load_button.pack(pady=5)

        self.k_label = tk.Label(root, text="Select Number of Clusters (K):")
        self.k_label.pack()
        self.k_value = ttk.Combobox(root, values=list(range(2, 11)), state="readonly")
        self.k_value.set(5)
        self.k_value.pack(pady=5)

        self.elbow_button = tk.Button(root, text="Show Elbow Curve", command=self.show_elbow, width=25)
        self.elbow_button.pack(pady=5)

        self.cluster_button = tk.Button(root, text="Run Clustering", command=self.cluster, width=25)
        self.cluster_button.pack(pady=5)
        
        self.silhouette_button = tk.Button(root, text="Evaluate Silhouette Score", command=self.show_silhouette, width=25)
        self.silhouette_button.pack(pady=5)


        self.summary_button = tk.Button(root, text="Show Cluster Summary", command=self.show_summary, width=25)
        self.summary_button.pack(pady=5)
        
        self.export_summary_button = tk.Button(root, text="Export Summary to CSV", command=self.export_summary, width=25)
        self.export_summary_button.pack(pady=5)


        self.save_button = tk.Button(root, text="Save Results to CSV", command=self.save_results, width=25)
        self.save_button.pack(pady=5)

        self.theme_button = tk.Checkbutton(root, text="Dark Mode", variable=self.is_dark, command=self.toggle_theme)
        self.theme_button.pack(pady=5)
        
        self.auto_k_button = tk.Button(root, text="Suggest Best K", command=self.suggest_best_k, width=25)
        self.auto_k_button.pack(pady=5)

        
        


        self.file_path = None
        self.df = None
        self.X = None
        self.rfm_df = None
        self.model = None
        self.labels = None

    def center_window(self, window, w, h):
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        x = (screen_w // 2) - (w // 2)
        y = (screen_h // 2) - (h // 2)
        window.geometry(f"{w}x{h}+{x}+{y}")

    def toggle_theme(self):
        if self.is_dark.get():
            self.root.configure(bg="#2e2e2e")
            for widget in self.root.winfo_children():
                try:
                    widget.configure(bg="#2e2e2e", fg="white")
                except:
                    pass
        else:
            self.root.configure(bg="SystemButtonFace")
            for widget in self.root.winfo_children():
                try:
                    widget.configure(bg="SystemButtonFace", fg="black")
                except:
                    pass

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            try:
                self.df = load_data(path)
                self.X = preprocess_data(self.df)
                self.file_path = path
                messagebox.showinfo("Success", f"Dataset loaded from:\n{os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def show_elbow(self):
        if self.X is not None:
            plot_elbow(self.X)
        else:
            messagebox.showwarning("Missing", "Please load a dataset first.")

    def cluster(self):
        if self.X is not None:
            k = int(self.k_value.get())
            self.model, self.labels = apply_kmeans(self.X, n_clusters=k)
            self.df['Cluster'] = self.labels
            plot_clusters(self.X, self.labels, self.model)
        else:
            messagebox.showwarning("Missing", "Please load a dataset first.")

    def show_summary(self):
        if self.df is not None and 'Cluster' in self.df.columns:
            if 'Annual Income (k$)' in self.df.columns and 'Spending Score (1-100)' in self.df.columns:
                summary = self.df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
                
                # Create a new window for summary charts
                summary_window = tk.Toplevel(self.root)
                summary_window.title("Cluster Summary - Charts")
                self.center_window(summary_window, 500, 300)

                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                summary['Annual Income (k$)'].plot(kind='bar', ax=ax[0], color='skyblue')
                ax[0].set_title("Avg Income by Cluster")
                ax[0].set_xlabel("Cluster")
                ax[0].set_ylabel("Income (k$)")

                summary['Spending Score (1-100)'].plot(kind='bar', ax=ax[1], color='salmon')
                ax[1].set_title("Avg Spending Score by Cluster")
                ax[1].set_xlabel("Cluster")
                ax[1].set_ylabel("Score")

                plt.tight_layout()

                canvas = FigureCanvasTkAgg(fig, master=summary_window)
                canvas.draw()
                canvas.get_tk_widget().pack(expand=True, fill='both')
            else:
                messagebox.showwarning("Missing Columns", "Dataset must include 'Annual Income (k$)' and 'Spending Score (1-100)'")
        else:
            messagebox.showwarning("Run Clustering", "Please run clustering first.")


    def save_results(self):
        if self.df is not None and self.labels is not None:
            result = self.df.copy()
            result['Cluster'] = self.labels
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if save_path:
                result.to_csv(save_path, index=False)
                messagebox.showinfo("Saved", f"Results saved to:\n{save_path}")
        else:
            messagebox.showwarning("Nothing to Save", "Run clustering first.")
            
    def export_summary(self):
        if self.df is not None and 'Cluster' in self.df.columns:
            if 'Annual Income (k$)' in self.df.columns and 'Spending Score (1-100)' in self.df.columns:
                summary = self.df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().reset_index()
                save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
                if save_path:
                    summary.to_csv(save_path, index=False)
                    messagebox.showinfo("Saved", f"Cluster summary saved to:\n{save_path}")
            else:
                messagebox.showwarning("Missing Columns", "Dataset must include 'Annual Income (k$)' and 'Spending Score (1-100)'")
        else:
            messagebox.showwarning("Run Clustering", "Please run clustering first.")
            
            
    

    def show_silhouette(self):
        if self.X is not None and self.labels is not None:
            try:
                score = silhouette_score(self.X, self.labels)
                messagebox.showinfo("Silhouette Score", f"Clustering Quality Score:\n{score:.3f}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to compute silhouette score:\n{str(e)}")
        else:
            messagebox.showwarning("Run Clustering", "Please run clustering first.")
            
    def suggest_best_k(self):
        if self.X is not None:
            from sklearn.cluster import KMeans
            import numpy as np

            distortions = []
            K = range(1, 11)
            for k in K:
                model = KMeans(n_clusters=k, random_state=42)
                model.fit(self.X)
                distortions.append(model.inertia_)

            # Use the "knee" method to suggest K
            from kneed import KneeLocator
            kl = KneeLocator(K, distortions, curve="convex", direction="decreasing")
            suggested_k = kl.elbow

            import matplotlib.pyplot as plt
            plt.plot(K, distortions, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k')
            plt.axvline(x=suggested_k, color='r', linestyle='--', label=f'Suggested k = {suggested_k}')
            plt.legend()
            plt.show()

            messagebox.showinfo("Suggested K", f"The optimal number of clusters (k) is likely:\n{suggested_k}")
        else:
            messagebox.showwarning("Missing Data", "Please load a dataset first.")




def run_gui():
    root = tk.Tk()
    app = KMeansGUI(root)
    root.mainloop()
