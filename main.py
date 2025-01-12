# Step 1: Set up the environment and import our class
import numpy as np
import matplotlib.pyplot as plt
from FacialRecognitionHDBSCAN import FacialRecognitionHDBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import sys
from pathlib import Path

def evaluate_clustering_metrics(face_clustering):
    """
    Evaluate clustering metrics: Precision, Recall, F1 score, and Accuracy.
    """
    try:
        ground_truth_labels = face_clustering.true_labels  # Adjust as per your implementation
        predicted_labels = face_clustering.cluster_labels

        # Filter out noise points (label -1)
        mask = predicted_labels != -1
        filtered_ground_truth = ground_truth_labels[mask]
        filtered_predicted = predicted_labels[mask]

        # Calculate metrics
        precision = precision_score(filtered_ground_truth, filtered_predicted, average='weighted', zero_division=0)
        recall = recall_score(filtered_ground_truth, filtered_predicted, average='weighted', zero_division=0)
        f1 = f1_score(filtered_ground_truth, filtered_predicted, average='weighted')
        accuracy = accuracy_score(filtered_ground_truth, filtered_predicted)

        print(f"\n=== Clustering Metrics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        return {'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy}
    except Exception as e:
        print(f"Error during metric evaluation: {e}")
        return None

# Increase recursion limit for large datasets
sys.setrecursionlimit(10000)

def run_facial_recognition_experiment(
        data_source="lfw",
        folder_path=None,
        min_faces_per_person=70,
        resize_factor=2,
        min_cluster_size=10,
        min_samples=5,
        add_noise=False,
        noise_factor=0.1,
        metrics=None
):
    """
    Run a complete facial recognition clustering experiment.
    """
    if metrics is None:
        metrics = ['euclidean']  # Default to 'euclidean' metric

    # Step 2: Initialize the clustering system
    face_clustering = FacialRecognitionHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    # Step 3: Load the data
    try:
        if data_source == "lfw":
            print("\n=== Loading LFW Dataset ===")
            face_clustering.load_lfw_dataset(
                min_faces_per_person=min_faces_per_person,
                resize=resize_factor
            )
        elif data_source == "local":
            if not folder_path or not Path(folder_path).exists():
                raise ValueError("Invalid or missing folder_path for local data source")
            print("\n=== Loading Local Dataset ===")
            face_clustering.load_from_folder(
                folder_path=folder_path,
                target_size=(64, 64)
            )
        else:
            raise ValueError("Unsupported data_source. Use 'lfw' or 'local'.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # Step 4: Preprocess the data
    print("\n=== Preprocessing Data ===")
    face_clustering.preprocess_data()

    # Optional: Add noise for robustness testing
    if add_noise:
        print("\n=== Adding Noise for Robustness Testing ===")
        face_clustering.add_noise(noise_factor=noise_factor)

    # Step 5: Perform clustering with different metrics
    results = {}

    for metric in metrics:
        print(f"\n=== Performing Clustering with {metric} metric ===")
        try:
            face_clustering.perform_clustering(metric=metric)

            # Step 6: Evaluate clustering
            print(f"\nEvaluation metrics for {metric}:")
            face_clustering.evaluate_clustering()

            # Step 7: Visualize results
            print(f"\nVisualizing clusters for {metric} metric...")
            face_clustering.visualize_clusters()

            # Visualize representatives
            print(f"\nVisualizing representative images for {metric} metric...")
            representatives = face_clustering.get_cluster_representatives()

            # Store results
            n_clusters = len(set(face_clustering.cluster_labels)) - (1 if -1 in face_clustering.cluster_labels else 0)
            results[metric] = {
                'labels': face_clustering.cluster_labels.copy(),
                'n_clusters': n_clusters
            }

            # Save model
            save_dir = f'{metric}'
            Path(save_dir).mkdir(exist_ok=True)
            face_clustering.save_model(save_dir)
            print(f"Model saved to: {save_dir}/")

            for cluster, idx in representatives.items():
                image = face_clustering.images[idx]
                plt.imshow(image, cmap='gray')
                plt.title(f"Cluster {cluster} Representative")
                plt.axis('off')
                plt.show()
        except Exception as e:
            print(f"Error during clustering with {metric}: {e}")

    return face_clustering, results

def visualize_additional_metrics(face_clustering):
    """
    Create additional visualizations to analyze the clustering results
    """
    try:
        if hasattr(face_clustering, 'clusterer') and face_clustering.clusterer is not None:
            # Plot cluster probabilities
            plt.figure(figsize=(10, 7))
            plt.hist(face_clustering.clusterer.probabilities_, bins=50)
            plt.title('Cluster Assignment Probabilities')
            plt.xlabel('Probability')
            plt.ylabel('Count')
            plt.show()

            # Plot 2D projection with cluster sizes
            plt.figure(figsize=(12, 8))
            unique_labels = np.unique(face_clustering.cluster_labels)
            for label in unique_labels:
                mask = face_clustering.cluster_labels == label
                plt.scatter(
                    face_clustering.features[mask, 0],
                    face_clustering.features[mask, 1],
                    label=f'Cluster {label}' if label != -1 else 'Noise',
                    alpha=0.7,
                    s=100
                )
            plt.title('Cluster Sizes in 2D Projection')
            plt.xlabel('First PCA Component')
            plt.ylabel('Second PCA Component')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # Print cluster statistics
            print("\nCluster Statistics:")
            for label in unique_labels:
                size = np.sum(face_clustering.cluster_labels == label)
                if label == -1:
                    print(f"Noise points: {size}")
                else:
                    print(f"Cluster {label} size: {size}")

    except Exception as e:
        print(f"Warning: Could not generate some visualizations due to: {str(e)}")

# Example usage
if __name__ == "__main__":
    config = {
        'data_source': 'lfw',
        'folder_path': None,
        'min_faces_per_person': 70,
        'resize_factor': 2,
        'min_cluster_size': 10,
        'min_samples': 5,
        'add_noise': False,
        'noise_factor': 0.05,
        'metrics': ['euclidean']
    }

    config1 = {
        'data_source': 'local',
        'folder_path': 'lfw',
        'min_faces_per_person': 70,
        'resize_factor': 5,
        'min_cluster_size': 2,
        'min_samples': 2,
        'add_noise': False,
        'noise_factor': 0.05,
        'metrics' : ['euclidean']
    }

    # Run the experiment
    clustering_model, results = run_facial_recognition_experiment(**config1)

    # Print summary
    if clustering_model and results:
        print("\n=== Experiment Summary ===")
        for metric, result in results.items():
            print(f"\nMetric: {metric}")
            print(f"Number of clusters: {result['n_clusters']}")
            print(f"Number of noise points: {list(result['labels']).count(-1)}")

        # Create additional visualizations
        visualize_additional_metrics(clustering_model)
