import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import joblib

# Function to load images from a local directory
def load_images_from_directory(directory, img_size=(50, 37)):
    images = []
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                if img is not None:
                    img_resized = cv2.resize(img, img_size)  # Resize to a fixed size
                    images.append(img_resized)
                    file_names.append(file)
    return np.array(images), file_names

# Function to fetch online dataset (LFW People)
def fetch_online_data():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=1)
    images = lfw_people.images
    labels = lfw_people.target
    target_names = lfw_people.target_names
    return images, labels, target_names

# Specify the dataset source: 'online' or 'offline'
data_source = 'offline'

if data_source == 'offline':
    print("Loading dataset locally...")
    image_directory = "./lfw"  # Update this path to the correct folder
    images, file_names = load_images_from_directory(image_directory)
    labels = None
    target_names = None
else:
    print("Fetching dataset online...")
    images, labels, target_names = fetch_online_data()
    file_names = target_names[labels]  # Assign meaningful names from the dataset

print(f"Number of Images: {len(images)}, Image Shape: {images[0].shape}")

# Visualize a few images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    if i < len(images):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
plt.show()

# Step 2: Data processing
flattened_data = images.reshape(len(images), -1)  # Flatten to (num_samples, num_features)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(flattened_data)

# Step 3: Dimensionality reduction
pca = PCA(n_components=0.95)  # Adjust components based on variance explained
reduced_data = pca.fit_transform(standardized_data)
print(f"Explained Variance Ratio: {sum(pca.explained_variance_ratio_):.2f}")

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Step 4: Clustering with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=5, metric='euclidean')
cluster_labels = clusterer.fit_predict(reduced_data)

num_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of Clusters: {num_clusters}")

# Step 5: Introducing real-life challenges
noisy_data = reduced_data + np.random.normal(0, 0.1, reduced_data.shape)  # Add Gaussian noise
noisy_labels = clusterer.fit_predict(noisy_data)

# Experiment distance metrics
clusterer_manhattan = hdbscan.HDBSCAN(min_cluster_size=10, metric='manhattan')
cluster_labels_manhattan = clusterer_manhattan.fit_predict(reduced_data)

# Step 6: Visualize and analyze results
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=5)
plt.colorbar()
plt.show()

noise_points = reduced_data[cluster_labels == -1]  # HDBSCAN labels noise as -1
print(f"Number of Noise Points: {len(noise_points)}")

# Calculate silhouette score
if num_clusters > 1:
    score = silhouette_score(reduced_data, cluster_labels)
    print(f"Silhouette Score: {score:.2f}")
else:
    print("Silhouette Score cannot be computed as there is only one cluster.")

# Step 6.5: Calculate additional metrics if ground truth labels are available
if labels is not None:
    valid_indices = cluster_labels != -1  # Exclude noise points
    accuracy = accuracy_score(labels[valid_indices], cluster_labels[valid_indices])
    precision = precision_score(labels[valid_indices], cluster_labels[valid_indices], average="weighted")
    recall = recall_score(labels[valid_indices], cluster_labels[valid_indices], average="weighted")
    f1 = f1_score(labels[valid_indices], cluster_labels[valid_indices], average="weighted")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

# Step 7: Save Results
print("Saving results...")
output_directory = "./clustering_results"  # Specify your desired directory
os.makedirs(output_directory, exist_ok=True)

# Save PCA-transformed data and cluster labels
pca_df = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(reduced_data.shape[1])])
pca_df["Cluster"] = cluster_labels
pca_df.to_csv(os.path.join(output_directory, "pca_clusters.csv"), index=False)
print("PCA and cluster labels saved to 'pca_clusters.csv'.")

# Save cluster metrics and details
cluster_details = {
    "Number of Clusters": num_clusters,
    "Noise Points": len(noise_points),
    "Silhouette Score": score if num_clusters > 1 else "Not applicable",
}
pd.DataFrame([cluster_details]).to_csv(os.path.join(output_directory, "cluster_metrics.csv"), index=False)
print("Cluster metrics saved to 'cluster_metrics.csv'.")

# Save noisy dataset and labels for further analysis
np.save(os.path.join(output_directory, "noisy_data.npy"), noisy_data)
np.save(os.path.join(output_directory, "noisy_labels.npy"), noisy_labels)
print("Noisy data and labels saved.")

# Save the trained model components
joblib.dump(scaler, "scaler.pkl")
joblib.dump(reduced_data, "reduced_training_data.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(clusterer, "clusterer.pkl")
print("Model components saved successfully!")
