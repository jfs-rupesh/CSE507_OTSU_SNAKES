import matplotlib.pyplot as plt
from PIL import Image
import snake as sn
import numpy as np

# Load grayscale image
image_path = 'coins.tif'  # Specify the path to your image
img = Image.open(image_path)  # Convert image to grayscale
img = np.array(img)  # Convert the image to a NumPy array

# Initialize snake parameters
t = np.arange(0, 2 * np.pi, 0.1)
x = 120 + 50 * np.cos(t)  # Initial x coordinates
y = 140 + 60 * np.sin(t)  # Initial y coordinates

# Snake parameters
alpha = 0.001
beta = 0.4
gamma = 100
iterations = 50

# External edge force gradients from the image
fx, fy = sn.create_external_edge_force_gradients_from_img(img)

# Iterate the snake
snakes = sn.iterate_snake(
    x=x,
    y=y,
    a=alpha,
    b=beta,
    fx=fx,
    fy=fy,
    gamma=gamma,
    n_iters=iterations,
    return_all=True
)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img, cmap=plt.cm.gray)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0], 0)
ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], c=(0, 1, 0), lw=2)

# Plot intermediate snakes
for i, snake in enumerate(snakes):
    if i % 10 == 0:
        ax.plot(np.r_[snake[0], snake[0][0]], np.r_[snake[1], snake[1][0]], c=(0, 0, 1), lw=2)

# Plot the final snake in red
ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1, 0, 0), lw=2)

plt.show()
