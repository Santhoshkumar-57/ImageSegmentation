# AI IMAGE COLOR SEGMENTATION USING GMM
# Adapted from the Google Colab version to run locally:
# - replaces google.colab files.upload/download with a local image path
# - saves all plots to the results/ folder instead of plt.show()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, silhouette_score
import os

INPUT_IMAGE = "input_flower.jpg"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. LOAD IMAGE
print("Loading image:", INPUT_IMAGE)
image = Image.open(INPUT_IMAGE).convert("RGB")

# Resize image for faster processing
image.thumbnail((180, 180))
original = np.array(image)

print("\nImage loaded successfully!")
print("Image dimensions:", original.shape)

# 2. PREPROCESSING
height, width, _ = original.shape
pixels = original.reshape(-1, 3).astype(np.float64)

# Randomly sample pixels for training
rng = np.random.default_rng(42)
sample_size = min(12000, len(pixels))
indices = rng.choice(len(pixels), sample_size, replace=False)
train_pixels = pixels[indices]

# 3. TRAIN GMM MODEL
k = 5

gmm = GaussianMixture(
    n_components=k,
    covariance_type="diag",
    random_state=42,
    reg_covar=1e-3,
    n_init=2,
    max_iter=100
)

gmm.fit(train_pixels)
print("\nGMM model trained successfully!")
print("Number of color groups:", k)
print("Model converged:", gmm.converged_)

# 4. PREDICT PIXEL CLUSTERS
labels = gmm.predict(pixels)
reconstructed = gmm.means_[labels]

segmented = np.clip(
    reconstructed.reshape(height, width, 3),
    0, 255
).astype(np.uint8)

# 5. SAVE ORIGINAL AND SEGMENTED IMAGES COMPARISON
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented)
plt.title("GMM Segmented Image (5 Groups)")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150)
plt.close()

# 6. SAVE LEARNED COLOR PALETTE
palette = np.clip(gmm.means_, 0, 255).astype(np.uint8)

plt.figure(figsize=(8, 2))
plt.imshow([palette])
plt.title("Learned GMM Color Palette")
plt.yticks([])
plt.xticks(range(k), [f"Group {i+1}" for i in range(k)])
plt.savefig(os.path.join(RESULTS_DIR, "palette.png"), dpi=150)
plt.close()

print("\nLearned RGB colors:")
for i, color in enumerate(palette, start=1):
    print(f"Group {i}: {tuple(color)}")

# 7. EVALUATION
mse = mean_squared_error(pixels, reconstructed)
print(f"\nReconstruction MSE: {mse:.2f}")

# Silhouette score on a sample
eval_size = min(1500, len(pixels))
eval_indices = rng.choice(len(pixels), eval_size, replace=False)
eval_pixels = pixels[eval_indices]
eval_labels = labels[eval_indices]

if 1 < len(np.unique(eval_labels)) < len(eval_labels):
    score = silhouette_score(eval_pixels, eval_labels)
    print(f"Silhouette Score: {score:.3f}")
else:
    print("Silhouette score unavailable for this sample.")

# 8. COMPARE DIFFERENT NUMBERS OF COLOR GROUPS
print("\nComparing GMM component counts...")

for components in [3, 5, 8, 10]:
    model = GaussianMixture(
        n_components=components,
        covariance_type="diag",
        random_state=42,
        reg_covar=1e-3,
        n_init=1,
        max_iter=100
    )
    model.fit(train_pixels)

    predicted_labels = model.predict(pixels)
    result = model.means_[predicted_labels]
    error = mean_squared_error(pixels, result)

    result_image = np.clip(
        result.reshape(height, width, 3),
        0, 255
    ).astype(np.uint8)

    print(f"Groups: {components} | MSE: {error:.2f}")

    plt.figure(figsize=(4, 4))
    plt.imshow(result_image)
    plt.title(f"GMM with {components} Groups")
    plt.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, f"gmm_{components}_groups.png"), dpi=150)
    plt.close()

# 9. SAVE RESULT
output_file = os.path.join(RESULTS_DIR, "gmm_segmented_image.png")
Image.fromarray(segmented).save(output_file)

print("\nProject completed successfully!")
print("Segmented image saved to:", output_file)
