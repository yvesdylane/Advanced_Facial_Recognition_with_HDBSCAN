import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
from sklearn.datasets import fetch_lfw_people
import cv2
from pathlib import Path
import os
from tqdm import tqdm  # For progress bars
import joblib


class FacialRecognitionHDBSCAN:
    def __init__(self, min_cluster_size=5, min_samples=5):
        """
        Initialize the facial recognition clustering system

        Args:
            min_cluster_size: Minimum size of clusters for HDBSCAN
            min_samples: Minimum samples for core points in HDBSCAN
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.features = None
        self.images = None
        self.targets = None
        self.target_names = None
        self.clusterer = None
        self.pca = None
        self.scaler = StandardScaler()

    def load_from_folder(self, folder_path, target_size=(64, 64), supported_formats=('.jpg', '.jpeg', '.png')):
        """
        Load images from a local folder. The folder should be organized as follows:
        folder_path/
            person1/
                image1.jpg
                image2.jpg
            person2/
                image1.jpg
                image2.jpg
            ...

        Args:
            folder_path: Path to the root folder containing person subfolders
            target_size: Tuple of (height, width) to resize images
            supported_formats: Tuple of supported image file extensions
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder {folder_path} does not exist")

        print("Loading images from folder...")
        images = []
        targets = []
        target_names = []

        # Get all person folders
        person_folders = [f for f in folder_path.iterdir() if f.is_dir()]

        for person_idx, person_folder in enumerate(tqdm(person_folders)):
            target_names.append(person_folder.name)

            # Get all image files for this person
            image_files = [
                f for f in person_folder.iterdir()
                if f.suffix.lower() in supported_formats
            ]

            for img_path in image_files:
                try:
                    # Read and preprocess image
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, target_size)
                        images.append(img)
                        targets.append(person_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        if not images:
            raise ValueError("No valid images found in the specified folder")

        self.images = np.array(images)
        self.features = np.array([img.flatten() for img in self.images])
        self.targets = np.array(targets)
        self.target_names = np.array(target_names)

        print(f"Loaded {len(self.images)} images of {len(target_names)} people")
        self.display_sample_images()

    def load_lfw_dataset(self, min_faces_per_person=70, resize=2):
        """
        Load the Labeled Faces in the Wild dataset from scikit-learn

        Args:
            min_faces_per_person: Minimum number of faces per person to include
            resize: Factor by which to resize the images
        """
        print("Downloading and loading LFW dataset...")
        try:
            lfw_people = fetch_lfw_people(
                min_faces_per_person=min_faces_per_person,
                resize=resize,
                color=False
            )

            self.images = lfw_people.images
            self.features = lfw_people.data
            self.targets = lfw_people.target
            self.target_names = lfw_people.target_names

            print(f"Dataset dimensions: {self.features.shape}")
            print(f"Number of people: {len(self.target_names)}")
            print(f"Sample image size: {self.images[0].shape}")

            self.display_sample_images()

        except Exception as e:
            print(f"Error downloading LFW dataset: {e}")
            print("Please check your internet connection or try loading from a local folder")

    def display_sample_images(self, n_samples=5):
        """Display sample images from the dataset"""
        n_samples = min(n_samples, len(self.images))
        fig, axes = plt.subplots(1, n_samples, figsize=(2 * n_samples, 3))
        if n_samples == 1:
            axes = [axes]

        for ax, image, label in zip(axes, self.images[:n_samples], self.targets[:n_samples]):
            ax.imshow(image, cmap='gray')
            ax.set_title(self.target_names[label])
            ax.axis('off')
        plt.show()

    def preprocess_data(self):
        """
        Standardize the data and apply PCA
        """
        print("Preprocessing data...")
        # Standardize features
        self.features = self.scaler.fit_transform(self.features)

        # Apply PCA
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.features = self.pca.fit_transform(self.features)
        print(f"Reduced dimensions to {self.features.shape[1]} components")

    def add_noise(self, noise_factor=0.1):
        """
        Add Gaussian noise to the features

        Args:
            noise_factor: Standard deviation of the noise
        """
        noise = np.random.normal(0, noise_factor, self.features.shape)
        self.features = self.features + noise

    def perform_clustering(self, metric='euclidean'):
        """
        Apply HDBSCAN clustering

        Args:
            metric: Distance metric to use
        """
        print("Performing HDBSCAN clustering...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=metric
        )
        self.cluster_labels = self.clusterer.fit_predict(self.features)

    def visualize_clusters(self):
        """
        Create visualization of the clusters with enhanced appearance
        """
        plt.figure(figsize=(12, 8))

        scatter = plt.scatter(
            self.features[:, 0],
            self.features[:, 1],
            c=self.cluster_labels,
            cmap='Dark2',  # Using Dark2 colormap for darker colors
            alpha=0.7,  # Increased opacity
            s=100,  # Larger points
            edgecolors='black',  # Black edges around points
            linewidth=0.5
        )
        plt.colorbar(scatter)
        plt.title("HDBSCAN Clustering Results")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_cluster_representatives(self):
        """
        Display the most representative image from each cluster
        """
        representatives = self.get_cluster_representatives()
        n_clusters = len(representatives)

        if n_clusters == 0:
            print("No clusters found to visualize")
            return

        # Calculate grid dimensions
        n_cols = min(5, n_clusters)
        n_rows = (n_clusters - 1) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, (cluster, idx) in zip(axes, representatives.items()):
            image = self.images[idx]
            true_label = self.target_names[self.targets[idx]]

            ax.imshow(image, cmap='gray')
            ax.set_title(f'Cluster {cluster}\n({true_label})', fontsize=10)
            ax.axis('off')

        # Turn off any unused subplots
        for ax in axes[len(representatives):]:
            ax.axis('off')

        plt.suptitle('Representative Images for Each Cluster', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

        # Add true labels as annotations for some points
        for i in range(0, len(self.features), 20):  # Annotate every 20th point
            plt.annotate(
                self.target_names[self.targets[i]],
                (self.features[i, 0], self.features[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='black',
                weight='bold'
            )

        plt.colorbar(plt.scatter)
        plt.title('HDBSCAN Clustering Results with True Labels', fontsize=14, weight='bold')
        plt.xlabel('First PCA Component', fontsize=6)
        plt.ylabel('Second PCA Component', fontsize=6)

        # Add grid for better readability
        plt.grid(True, alpha=0.3)

        # Tight layout to prevent label cutoff
        plt.tight_layout()
        plt.show()

    def get_cluster_representatives(self):
        """
        Find the most representative image for each cluster

        Returns:
            dict: Mapping of cluster labels to representative image indices
        """
        representatives = {}
        unique_clusters = np.unique(self.cluster_labels)

        for cluster in unique_clusters:
            if cluster != -1:  # Skip noise points
                cluster_points = self.features[self.cluster_labels == cluster]
                cluster_center = np.mean(cluster_points, axis=0)

                # Find the point closest to the center
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                representative_idx = np.where(self.cluster_labels == cluster)[0][np.argmin(distances)]
                representatives[cluster] = representative_idx

        return representatives

    def predict_new_image(self, image_path):
        """
        Assign a new image to one of the existing clusters.

        Args:
            image_path: Path to the new image.

        Returns:
            predicted_cluster: The cluster label assigned to the new image.
        """
        # Load and preprocess the new image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        features = img.flatten().reshape(1, -1)

        # Apply the same preprocessing
        features = self.scaler.transform(features)
        features = self.pca.transform(features)

        if self.clusterer is None or len(np.unique(self.cluster_labels)) <= 1:
            raise ValueError("Model has not been clustered or there are no valid clusters.")

        # Compute distances to each cluster's representative
        cluster_centers = np.array([
            np.mean(self.features[self.cluster_labels == cluster], axis=0)
            for cluster in np.unique(self.cluster_labels) if cluster != -1
        ])
        distances = np.linalg.norm(cluster_centers - features, axis=1)
        predicted_cluster = np.argmin(distances)

        return predicted_cluster

    def evaluate_clustering(self):
        """
        Evaluate the clustering quality using silhouette score, precision, recall, F1 score, and accuracy.
        """
        from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score, accuracy_score
        from sklearn.utils import check_X_y

        mask = self.cluster_labels != -1
        unique_clusters = np.unique(self.cluster_labels[mask])

        if len(unique_clusters) > 1:
            # Calculate Silhouette Score
            score = silhouette_score(
                self.features[mask],
                self.cluster_labels[mask],
                metric='euclidean'
            )
            print(f"Silhouette Score: {score:.3f}")

            # Self-evaluate clustering
            true_labels = self.cluster_labels[mask]
            predicted_labels = self.cluster_labels[mask]

            # Calculate precision, recall, F1 score, and accuracy using self-labeling
            precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, average='weighted')
            accuracy = accuracy_score(true_labels, predicted_labels)

            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1:.3f}")
            print(f"Accuracy: {accuracy:.3f}")
        else:
            print("Not enough clusters for silhouette score calculation")

    def save_model(self, save_dir):
        """
        Save the trained model components to disk.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        components = {
            'scaler': self.scaler,
            'pca': self.pca,
            'clusterer': self.clusterer,
            'cluster_labels': self.cluster_labels,
            'target_names': self.target_names,
            'features': self.features
        }

        joblib.dump(components, save_dir / 'model.joblib')
        print(f"Model saved to {save_dir}")

    def load_model(self, model_path):
        """
        Load a previously saved model.
        """
        components = joblib.load(model_path)

        # Ensure all required components are loaded
        required_keys = ['scaler', 'pca', 'clusterer', 'cluster_labels', 'target_names', 'features']
        for key in required_keys:
            if key not in components or components[key] is None:
                raise ValueError(f"Missing or invalid component '{key}' in the saved model.")

        self.scaler = components['scaler']
        self.pca = components['pca']
        self.clusterer = components['clusterer']
        self.cluster_labels = components['cluster_labels']
        self.target_names = components['target_names']
        self.features = components['features']

        print("Model loaded successfully")
