import cv2
import joblib
import numpy as np

# Load the trained model components
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
clusterer = joblib.load("clusterer.pkl")
training_reduced_data = joblib.load("reduced_training_data.pkl")  # Load reduced training data
print("Model components loaded successfully!")

# Preprocess image
def preprocess_image(image, img_size=(50, 37)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, img_size)
    flattened_image = resized_image.flatten()
    return flattened_image

# Predict the cluster using the nearest cluster approach
def predict_with_hdbscan(new_data, clusterer, reduced_data):
    if not hasattr(clusterer, 'labels_'):
        raise ValueError("The clusterer does not have fitted labels. Please train it first.")

    # Compute cluster centers based on reduced_data and the clusterer labels
    cluster_centers = []
    for label in np.unique(clusterer.labels_):
        if label != -1:  # Exclude noise points
            cluster_points = reduced_data[clusterer.labels_ == label]
            cluster_centers.append(cluster_points.mean(axis=0))

    cluster_centers = np.array(cluster_centers)

    # Assign the new data point to the closest cluster center
    distances = np.linalg.norm(cluster_centers - new_data, axis=1)
    closest_cluster = np.argmin(distances)
    return closest_cluster

# Process and predict the cluster
def process_and_predict(image, scaler, pca, clusterer, training_reduced_data):
    preprocessed = preprocess_image(image)
    standardized = scaler.transform([preprocessed])
    reduced = pca.transform(standardized)

    # Predict the cluster
    cluster = predict_with_hdbscan(reduced, clusterer, training_reduced_data)
    return cluster, reduced

# Predict from an image file
def predict_from_file(file_path, scaler, pca, clusterer, training_reduced_data):
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image. Please check the file path.")
        return

    cluster, reduced_data = process_and_predict(image, scaler, pca, clusterer, training_reduced_data)
    print(f"The image belongs to cluster {cluster}.")
    return cluster

# Predict using the webcam
def predict_from_webcam(scaler, pca, clusterer, training_reduced_data):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to access the webcam.")
        return

    print("Press 's' to capture an image and predict its cluster, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cluster, reduced_data = process_and_predict(frame, scaler, pca, clusterer, training_reduced_data)
            print(f"The captured image belongs to cluster {cluster}.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main entry point
if __name__ == "__main__":
    mode = input("Choose mode: 'file' to test with an image file, 'webcam' to use your webcam: ").strip().lower()

    if mode == "file":
        file_path = input("Enter the path to the image file: ").strip()
        predict_from_file(file_path, scaler, pca, clusterer, training_reduced_data)
    elif mode == "webcam":
        predict_from_webcam(scaler, pca, clusterer, training_reduced_data)
    else:
        print("Invalid mode. Please choose 'file' or 'webcam'.")
