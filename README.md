# AI Image Color Segmentation Using Gaussian Mixture Models (GMM)

## Project Overview

This mini project implements **Image Color Segmentation using Gaussian Mixture Models (GMM)**, an unsupervised machine learning algorithm. It groups similar colors in an image into a specified number of color clusters and replaces each pixel with its cluster's representative color.

The project uses a flower image as input and generates segmented images with 3, 5, 8, and 10 color groups. It also evaluates the reconstruction quality and visualizes the learned color palette.

## Objectives

* Understand image color segmentation using unsupervised machine learning.
* Apply Gaussian Mixture Models to RGB pixel data.
* Reduce the number of colors in an image.
* Compare segmentation results using different numbers of color groups.
* Evaluate reconstruction quality using Mean Squared Error (MSE) and Silhouette Score.

## Technologies Used

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn
* **Numerical Processing:** NumPy
* **Image Processing:** Pillow (PIL)
* **Data Visualization:** Matplotlib
* **Algorithm:** Gaussian Mixture Model (GMM)

## How It Works

1. **Image Loading:** Loads the input flower image and converts it to RGB format.
2. **Preprocessing:** Resizes the image to a maximum dimension of 180 × 180 pixels and converts it into a collection of RGB pixels.
3. **Pixel Sampling:** Randomly selects up to 12,000 pixels for model training.
4. **Model Training:** Trains a GMM with 5 Gaussian components to identify groups of similar colors.
5. **Color Segmentation:** Assigns each pixel to a color group and replaces it with the corresponding learned color.
6. **Visualization:** Generates an original-versus-segmented comparison, a learned color palette, and segmentation results for different component counts.
7. **Evaluation:** Calculates MSE and Silhouette Score to assess reconstruction error and cluster separation.

## Project Structure

```text
ImageSegmentation_AI/
│
├── gmm_color_segmentation/
│   ├── code.py
│   ├── input_flower.jpg
│   └── results/
│       ├── comparison.png
│       ├── palette.png
│       ├── gmm_3_groups.png
│       ├── gmm_5_groups.png
│       ├── gmm_8_groups.png
│       ├── gmm_10_groups.png
│       └── gmm_segmented_image.png
│
├── gmm_color_segmentation.zip
└── README.md
```

## Installation

Make sure Python 3.9 or later is installed.

Install the required libraries using:

```bash
pip install numpy matplotlib pillow scikit-learn
```

## How to Run

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Ensure `code.py` and `input_flower.jpg` are in the same directory.
4. Open the terminal in the `gmm_color_segmentation` folder.
5. Run the following command:

```bash
python code.py
```

The program processes the image, evaluates the model, and saves the output images in the `results/` directory.

## Output Files

| File                      | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `comparison.png`          | Original image compared with the segmented image |
| `palette.png`             | Five representative colors learned by GMM        |
| `gmm_3_groups.png`        | Segmentation using 3 color groups                |
| `gmm_5_groups.png`        | Segmentation using 5 color groups                |
| `gmm_8_groups.png`        | Segmentation using 8 color groups                |
| `gmm_10_groups.png`       | Segmentation using 10 color groups               |
| `gmm_segmented_image.png` | Final segmented output image                     |

## Evaluation Metrics

* **Mean Squared Error (MSE):** Measures the average squared difference between original RGB pixel values and reconstructed pixel values. Lower values indicate less reconstruction error.
* **Silhouette Score:** Measures how well-separated the predicted pixel clusters are. Higher values generally indicate better cluster separation.

## Applications

* Image simplification and color reduction.
* Image preprocessing for computer vision.
* Image analysis and object segmentation research.
* Color-based image grouping.
* Educational demonstrations of unsupervised machine learning.

## Limitations

* The model groups pixels based on RGB color values and does not explicitly consider spatial relationships between neighboring pixels.
* The results depend on the selected number of Gaussian components.
* Similar colors belonging to different objects may be grouped together.
* Processing larger images may require additional computation.

## Future Improvements

* Allow users to upload their own images.
* Add an interactive interface using Streamlit.
* Compare GMM with K-Means clustering.
* Experiment with different color spaces, such as HSV and Lab.
* Explore spatial information to improve object segmentation.

## Conclusion

This project demonstrates how Gaussian Mixture Models can perform unsupervised image color segmentation by grouping RGB pixels into representative color clusters. It provides visual comparisons, learned color palettes, and evaluation metrics to help understand the relationship between the number of color groups and image reconstruction quality.

---

**Project Type:** Mini Project
**Domain:** Machine Learning and Computer Vision
**Algorithm:** Gaussian Mixture Model (GMM)
