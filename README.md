# Advanced Facial Recognition Clustering using HDBSCAN 🙌🏼

## Author
**DONFACK TSOPFACK YVES DYLANE**

---

## Introduction 🌎
Welcome to the GitHub repository for **Advanced Facial Recognition Clustering using HDBSCAN**!
Facial recognition plays a pivotal role in modern AI applications, yet clustering facial data in unsupervised settings presents challenges like noisy datasets, high-dimensional data, and unbalanced distributions. This project explores how HDBSCAN, a cutting-edge clustering algorithm, addresses these challenges effectively.

With its capability to handle noise and clusters of varying densities, HDBSCAN makes an excellent choice for real-world facial recognition datasets.

---

## Objective 🎯
### Key Goals:
- 🔎 Apply HDBSCAN to cluster facial images in an unsupervised learning scenario.
- 🏃️‍♂️ Address challenges such as noisy and unbalanced datasets.
- 🔬 Visualize and evaluate clustering results for practical implications.

---

## Prerequisites 🔧
### Knowledge Requirements:
- 🧠 Familiarity with unsupervised learning concepts and clustering algorithms.
- 📊 Understanding of dimensionality reduction techniques (PCA).
- 🐍 Proficiency in Python.

### Tools and Libraries Used:
- 💻 **Development Environment:** VS Code
- 📚 **Libraries:** NumPy, Pandas, Scikit-learn, OpenCV, Matplotlib, HDBSCAN

---

## Dataset 🔍
- 📂 **Source:** [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
- 🖼️ **Content:** 13,233 facial images representing 5,749 individuals.
- 🛠️ **Preprocessing:** Images resized to **50x37 pixels** and converted to grayscale for consistency.

---

## Methodology 🔬
### Steps Overview:
1. **Setup the Environment**
   - 🔌 Import necessary libraries like NumPy, Matplotlib, and HDBSCAN.
2. **Load and Explore Dataset**
   - 📥 Load the dataset locally or online.
   - 🖼️ Visualize samples to assess quality.
3. **Data Preprocessing**
   - 📏 Flatten images into 1D arrays and standardize using `StandardScaler`.
4. **Dimensionality Reduction**
   - 📉 Apply PCA to retain 95% variance while reducing dimensionality.
5. **Apply HDBSCAN Clustering**
   - ⚙️ Configure metrics like Euclidean and Manhattan distances.
   - 📊 Visualize clusters and analyze results.
6. **Introduce Real-Life Challenges**
   - 🌐 Simulate noisy data and test robustness.
7. **Visualize and Analyze Results**
   - 📈 Evaluate clustering using metrics like Silhouette Score, Precision, and Recall.
8. **Real-Life Applications**
   - 🧩 Assign new facial images to clusters using the trained model.

---

## Installation & Setup 🚀
1. 🖥️ Clone this repository:
   ```bash
   git clone https://github.com/yvesdylane/Advanced_Facial_Recognition_with_HDBSCAN
   ```
2. 📂 Navigate to the project directory:
   ```bash
   cd repo-name
   ```
3. 📦 Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. 🔨 Build the project:
   ```bash
   python main.py
   ```
   > **🚫 Note:** Building the model is a prerequisite for testing!

---

## Testing the Model 🔢
To test the clustering model:
1. ✅ Ensure the model is built (see Installation & Setup).
2. ▶️ Run the script for testing:
   ```bash
   python test_model.py
   ```
3. 🎥 Choose an input method:
   - 📹 **Webcam** for live testing.
   - 🖼️ **Image Path** for pre-stored images.
4. 🧐 Analyze the results! The script will display the assigned cluster.

---

## Results 🏆
### Key Metrics:
- 🏅 **Precision, Recall, F1 Score, Accuracy:** Perfect at **1.000**.
- 📏 **Silhouette Score:** 0.187 (indicates potential challenges in cluster separation).
- 🌀 **Clusters Identified:** Varies depending on noise and parameters.

### Visualizations:
- 📈 Scatter plots for clusters in reduced dimensions.
- 🖼️ Representative images for each cluster.

---

## Fun Insights 😂
- 💨 Clustering noisy data? HDBSCAN handles it like a champ!
- 🔨 Choose your distance metric wisely—Euclidean and Manhattan give different vibes!
- 😮‍♂️ Silhouette Scores can be tricky—don’t let them fool you!

---

## Future Work 🔄
- 🗂️ Test HDBSCAN with larger, more diverse datasets.
- 🤖 Integrate deep learning-based feature extraction methods.
- ⚙️ Optimize algorithm parameters for better cluster separation.

---

## Contributing 🤝
Contributions are welcome! Feel free to fork this repo and submit pull requests.

---

## License 📢
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments 🙏
Special thanks to the creators of the [LFW Dataset](http://vis-www.cs.umass.edu/lfw/) and the developers of HDBSCAN!

---

## Connect 🚀
Stay in touch and follow updates:
- 🌐 GitHub: [[GitHub Link](https://github.com/yvesdylane/Advanced_Facial_Recognition_with_HDBSCAN)]

Happy clustering! 🎉

