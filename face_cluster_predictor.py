import cv2
import numpy as np
from pathlib import Path
import sys
from FacialRecognitionHDBSCAN import FacialRecognitionHDBSCAN


def process_image(image, model):
    """Process image and predict cluster."""
    # Validate the model
    if model.features is None or model.cluster_labels is None or model.clusterer is None:
        raise ValueError("Model has not been trained or clusters are not available.")

    # Validate the input image
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for prediction.")

    # Resize and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (64, 64))

    # Flatten and preprocess
    features = resized.flatten().reshape(1, -1)
    features = model.scaler.transform(features)
    features = model.pca.transform(features)

    # Compute distances to cluster centroids
    cluster_centers = np.array([
        np.mean(model.features[model.cluster_labels == cluster], axis=0)
        for cluster in np.unique(model.cluster_labels) if cluster != -1
    ])
    if cluster_centers.size == 0:
        raise ValueError("No valid clusters available to make predictions.")

    distances = np.linalg.norm(cluster_centers - features, axis=1)
    predicted_cluster = np.argmin(distances)

    return predicted_cluster


def main():
    # Load the model
    model_path = input("Enter path to model file: ")
    if not Path(model_path).exists():
        print("Model file not found!")
        return

    model = FacialRecognitionHDBSCAN()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Ask for input method
    method = input("Choose input method (1 for webcam, 2 for local image): ")

    if method == "1":
        # Webcam mode
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam!")
            return

        print("Press 'S' to capture an image or 'Q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            cv2.imshow('Press S to capture, Q to quit', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                try:
                    # Process frame and predict cluster
                    cluster = process_image(frame, model)
                    print(f"Predicted Cluster: {cluster}")

                    # Display representative if available
                    representatives = model.get_cluster_representatives()
                    if cluster in representatives:
                        rep_idx = representatives[cluster]
                        rep_image = model.images[rep_idx]
                        true_label = model.target_names[model.targets[rep_idx]]

                        cv2.imshow('Cluster Representative', rep_image)
                        print(f"Matched with cluster representative: {true_label}")
                    else:
                        print("No representative found for the predicted cluster.")
                except Exception as e:
                    print(f"Error during prediction: {e}")

        cap.release()
        cv2.destroyAllWindows()

    elif method == "2":
        # Local image mode
        image_path = input("Enter path to image: ")
        if not Path(image_path).exists():
            print("Image not found!")
            return

        image = cv2.imread(image_path)
        if image is None:
            print("Cannot read image!")
            return

        try:
            # Process image and predict cluster
            cluster = process_image(image, model)
            print(f"Predicted Cluster: {cluster}")

            # Display original image and representative
            cv2.imshow('Input Image', image)

            representatives = model.get_cluster_representatives()
            if cluster in representatives:
                rep_idx = representatives[cluster]
                rep_image = model.images[rep_idx]
                true_label = model.target_names[model.targets[rep_idx]]

                cv2.imshow('Cluster Representative', rep_image)
                print(f"Matched with cluster representative: {true_label}")
            else:
                print("No representative found for the predicted cluster.")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during prediction: {e}")

    else:
        print("Invalid input method!")

if __name__ == "__main__":
    main()
