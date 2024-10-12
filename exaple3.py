import numpy as np
import matplotlib.pyplot as plt
import snake as sn
from skimage import measure, filters
from skimage.io import imread

# Read the image
image = imread('coins.tif')

# Convert to grayscale if necessary
if image.ndim == 3:
    image = np.mean(image, axis=2)  # Simple average for RGB to grayscale

# Threshold the image to create a binary mask
thresh = filters.threshold_otsu(image)
binary_mask = image > thresh

# Label connected components (coins)
labeled_coins = measure.label(binary_mask)

# Parameters for the snake
alpha = 0.001
beta = 0.01
gamma = 100
iterations = 50

# Prepare the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image, cmap='gray')
ax.set_xticks([])
ax.set_yticks([])

# Iterate through each labeled coin and apply the snake algorithm
for coin_label in np.unique(labeled_coins):
    if coin_label == 0:
        continue  # Skip background
    # Get the coordinates of the coin
    coin_mask = labeled_coins == coin_label
    y_coords, x_coords = np.where(coin_mask)

    # Create an initial snake around the coin's centroid
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    t = np.linspace(0, 2 * np.pi, 100)
    x_snake = centroid_x + 20 * np.cos(t)
    y_snake = centroid_y + 20 * np.sin(t)

    # Create external force fields
    fx, fy = sn.create_external_edge_force_gradients_from_img(image, sigma=10)

    # Run the snake algorithm
    snakes = sn.iterate_snake(
        x=x_snake,
        y=y_snake,
        a=alpha,
        b=beta,
        fx=fx,
        fy=fy,
        gamma=gamma,
        n_iters=iterations,
        return_all=True
    )

    # Plot all iterations of the snake
    for i, snake in enumerate(snakes):
        if i % 10 == 0:
            ax.plot(np.r_[snake[0], snake[0][0]], np.r_[snake[1], snake[1][0]], c=(0, 0, 1), lw=2)

    # Plot the last iteration in red
    ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1, 0, 0), lw=2)

plt.title("Active Contour Segmentation of Multiple Coins")
plt.show()
