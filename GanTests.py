import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.datasets import mnist
from scipy.linalg import sqrtm

# Load MNIST data
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0  # Normalize to [0, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension

# Define a simple generator
def create_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Create the generator
generator = create_generator()

# Load pre-trained InceptionV3 model
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# Function to preprocess images for Inception model
def preprocess_images(images):
    images = tf.image.resize(images, (299, 299))  # Resize images
    images = tf.image.grayscale_to_rgb(images)  # Convert grayscale to RGB
    images = preprocess_input(images)  # Preprocess for Inception
    return images

# Function to calculate FID
def calculate_fid(real_images, generated_images):
    # Preprocess images
    real_images = preprocess_images(real_images)
    generated_images = preprocess_images(generated_images)

    # Calculate activations
    real_activations = inception_model.predict(real_images, batch_size=100)
    generated_activations = inception_model.predict(generated_images, batch_size=100)

    # Calculate mean and covariance
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Function to calculate IS
def calculate_is(generated_images):
    generated_images = preprocess_images(generated_images)
    p_yx = inception_model.predict(generated_images, batch_size=100)

    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    kl_d = p_yx * (np.log(p_yx + 1e-16) - np.log(p_y + 1e-16))
    is_score = np.exp(np.mean(np.sum(kl_d, axis=1)))

    return is_score

# Generate images
noise = tf.random.normal([1000, 100])
generated_images = generator(noise, training=False).numpy()

# Calculate FID and IS
fid = calculate_fid(x_train[:1000], generated_images)
is_score = calculate_is(generated_images)

print(f"FID Score: {fid:.4f}")
print(f"Inception Score: {is_score:.4f}")

# Plot original and generated images

def plot_images(original, generated, num_images=10, filename='generated_images.png'):
    plt.figure(figsize=(12, 6))

    # Plot original images
    plt.subplot(2, num_images, 1)
    plt.title("Original MNIST Digits")
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    # Plot generated images
    plt.subplot(2, num_images, num_images + 1)
    plt.title("Generated MNIST Digits")
    for i in range(num_images):
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow((generated[i] + 1) / 2, cmap='gray')  # Scale from [-1, 1] to [0, 1]
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)  # Save the figure
    plt.close()  # Close the plot to free memory

# Plotting a sample of original and generated images

plot_images(x_train[:10], generated_images[:10], filename='generated_images.png')

