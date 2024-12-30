
import cv2
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def calculate_clustering_metrics(img_data, max_k=8):
    """
    Calculate clustering metrics for Elbow, Silhouette, and BIC methods.
    
    Parameters:
        img_data (np.ndarray): Image data (reshaped).
        max_k (int): Maximum number of clusters to test.
    
    Returns:
        dict: A dictionary containing inertia, silhouette scores, and BIC scores for each k.
    """
    inertia = []
    silhouette_scores = []
    bic_scores = []

    for k in range(2, max_k + 1):  # Start from 2 clusters since silhouette needs at least 2
        print(f"Calculating metrics for k = {k}...")
        
        # KMeans for inertia and silhouette
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(img_data)
        
        # Append inertia (Elbow Method)
        inertia.append(kmeans.inertia_)
        
        # Append silhouette score (subsample for speed if dataset is large)
        sample_data = img_data[np.random.choice(img_data.shape[0], size=min(1000, img_data.shape[0]), replace=False)]
        silhouette_scores.append(silhouette_score(sample_data, kmeans.labels_[:sample_data.shape[0]]))
        
        # Gaussian Mixture for BIC
        gmm = GaussianMixture(n_components=k, random_state=0)
        gmm.fit(img_data)
        bic_scores.append(gmm.bic(img_data))
    
    return {
        "inertia": inertia,
        "silhouette_scores": silhouette_scores,
        "bic_scores": bic_scores
    }

def determine_optimal_clusters(metrics):
    """
    Provide recommendations for optimal clusters based on the metrics.
    
    Parameters:
        metrics (dict): A dictionary containing clustering metrics.
    
    Returns:
        int: Optimal number of clusters based on user input.
    """
    silhouette_scores = metrics["silhouette_scores"]
    bic_scores = metrics["bic_scores"]

    print("\nRecommendations:")
    print(f" - Based on Silhouette, optimal k = {np.argmax(silhouette_scores) + 2}")
    print(f" - Based on BIC, optimal k = {np.argmin(bic_scores) + 2}")

    # Validate user input
    while True:
        try:
            #optimal_k = int(input("Enter the optimal number of clusters based on the analysis: "))
            optimal_k = 8
            if 2 <= optimal_k <= len(silhouette_scores) + 1:
                break
            else:
                print(f"Please enter a number between 2 and {len(silhouette_scores) + 1}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    
    return optimal_k


def display_clustering_metrics(metrics, max_k=10):
    """
    Display the clustering metrics using matplotlib plots.
    
    Parameters:
        metrics (dict): A dictionary containing clustering metrics.
        max_k (int): Maximum number of clusters to test.
    """
    inertia = metrics["inertia"]
    silhouette_scores = metrics["silhouette_scores"]
    bic_scores = metrics["bic_scores"]

   

    # Plot Silhouette Analysis
    plt.subplot(1, 3, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', label="Silhouette Score", color="green")
    plt.title("Silhouette Analysis")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.legend()

    # Plot BIC Scores
    plt.subplot(1, 3, 3)
    plt.plot(range(2, max_k + 1), bic_scores, marker='o', label="BIC", color="red")
    plt.title("BIC Scores")
    plt.xlabel("Number of Clusters")
    plt.ylabel("BIC (Lower is Better)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



def hex_to_rgb(hex_color):
    """
    Convert a hex color string to an RGB tuple.
    
    Parameters:
        hex_color (str): Hex color code (e.g., "#FFFFFF").
    
    Returns:
        tuple: RGB color as a tuple (R, G, B).
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def kmeans_discretize(image_path, n_colors=None, key_colors=None):
    """
    Discretize an image into `n_colors` using K-means clustering, ensuring that key colors are preserved.

    Parameters:
        image_path (str): Path to the input image.
        n_colors (int): Total number of colors to quantize the image into.
        key_colors (list): List of hex color codes to be preserved in the clustering.

    Returns:
        np.ndarray: Discretized image in RGB format.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unable to read at path: {image_path}")

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_data = img_rgb.reshape((-1, 3))

    # Handle case where n_colors is None
    if n_colors is None:
        # Calculate metrics and determine optimal clusters
        metrics = calculate_clustering_metrics(img_data, max_k=10)
        display_clustering_metrics(metrics, max_k=10)
        print("metrics done \n")
        n_colors = determine_optimal_clusters(metrics)
        print(f"Optimal number of clusters: {n_colors}")

    # If no key colors are provided, use the standard KMeans method
    if not key_colors:
        kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init="auto")
        kmeans.fit(img_data)
        new_img_data = kmeans.cluster_centers_[kmeans.labels_]

        # Round the cluster centers to the nearest integer to ensure distinct RGB values
        new_img_data = np.round(new_img_data).astype(np.uint8)

        # Reshape the result back to the original image shape
        discretized_img = new_img_data.reshape(img_rgb.shape)

        return discretized_img

    # Convert key colors from hex to RGB
    key_colors_rgb = [hex_to_rgb(color) for color in key_colors]

    if n_colors < len(key_colors_rgb):
        raise ValueError("The number of colors (n_colors) must be greater than or equal to the number of key colors.")

    # Initialize KMeans clustering for remaining colors
    remaining_clusters = n_colors - len(key_colors_rgb)
    additional_centers = np.empty((0, 3))  # Default for no additional clusters

    if remaining_clusters > 0:
        kmeans = KMeans(n_clusters=remaining_clusters, random_state=0, n_init="auto")
        kmeans.fit(img_data)
        additional_centers = kmeans.cluster_centers_

    # Combine key colors with cluster centers
    combined_centers = np.vstack((key_colors_rgb, additional_centers))

    # Assign each pixel to the nearest cluster center
    labels = np.argmin(np.linalg.norm(img_data[:, None] - combined_centers, axis=2), axis=1)
    new_img_data = combined_centers[labels]

    # Round the cluster centers to the nearest integer to ensure distinct RGB values
    new_img_data = np.round(new_img_data).astype(np.uint8)

    # Reshape the result back to the original image shape
    discretized_img = new_img_data.reshape(img_rgb.shape)

    # Convert the image back to BGR for saving with OpenCV
    discretized_img_bgr = cv2.cvtColor(discretized_img, cv2.COLOR_RGB2BGR)

    return discretized_img


def save_discretized_image(image, input_path, n_colors, method):
    """
    Save the discretized image with a modified file name, ensuring it is saved as PNG.
    
    Parameters:
        image (PIL.Image.Image): The discretized image.
        input_path (str): The input image file path.
        n_colors (int): Number of colors in the discretized image.
        method (str): The method used for discretization (e.g., "median_cut").
        output_folder (str): The folder where the image will be saved.
    """
    # Ensure the image is a PIL Image object
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)  # Convert from NumPy array to PIL Image
    
    # Extract the directory, base name, and extension
    directory, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    
    # Create the output file name with .png extension
    output_filename = f"{name}_discretized_{n_colors}_{method}.png"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save the image as PNG
    image.save(output_path, "PNG")
    print(f"Discretized image saved to {output_path}")

def display_images_side_by_side(original_image_path, discretized_image, n_colors, method):
    """
    Display the original and discretized images side by side.
    
    Parameters:
        original_image_path (str): Path to the original image.
        discretized_image (PIL.Image.Image): The discretized image.
        n_colors (int): Number of colors in the discretized image.
        method (str): The method used for discretization (e.g., "median_cut").
    """
    # Load the original image
    original_image = Image.open(original_image_path)
    
    # Display images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(discretized_image)
    plt.title(f"Discretized Image ({n_colors} Colors, {method})")
    plt.axis("off")
    
    plt.show()


# Function to separate individual colors from the image and return separate PNG images
def separate_colors(image_path, n_colors=None, output_dir=None, precision=0):
    """
    Separate the individual colors from the input image and return PNG images for each unique color.
    
    :param image_path: Path to the input image
    :param n_colors: Number of colors to consider (optional, uses all unique colors if None)
    :param output_dir: Directory to save the separated PNG images (optional)
    :param precision: Number of decimal places to round colors (default: 0, which rounds to integers)
    :return: None (Saves the PNG images)
    """
    # Open the image and convert to RGBA (if not already)
    img = Image.open(image_path).convert("RGBA")
    img_data = np.array(img)

    # If output directory is not specified, use the same directory as the input image
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # Identify the unique colors in the image (considering RGB channels only)
    # Round the colors to remove small differences (if any)
    rgb_data = img_data[:, :, :3].reshape(-1, 3)  # Flatten image data to 2D array
    if precision > 0:
        rgb_data = np.round(rgb_data / (10 ** precision)) * (10 ** precision)  # Round to specified precision
    unique_colors = np.unique(rgb_data, axis=0)
    
    # If n_colors is specified, limit the number of unique colors
    if n_colors is not None:
        unique_colors = unique_colors[:n_colors]
    
    # Debug: Print unique colors detected
    print(f"Detected unique colors: {unique_colors}")
    
    # Create an output folder for the PNG images
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = os.path.join(output_dir, base_name + "_separated_colors")
    os.makedirs(output_folder, exist_ok=True)

    # Process each color and create a new image where only that color is visible
    for idx, color in enumerate(unique_colors):
        # Create a mask for the current color (reshaped to match the original image shape)
        mask = np.all(rgb_data == color, axis=-1)
        
        # Ensure the mask has the same shape as the image data
        mask_reshaped = mask.reshape(img_data[:, :, 0].shape)

        # Create a new image where the current color is visible and others are transparent
        new_img_data = np.zeros_like(img_data)
        new_img_data[mask_reshaped] = img_data[mask_reshaped]  # Keep the color where the mask is true

        # Set the rest to be transparent
        new_img_data[~mask_reshaped, 3] = 0  # Make the other parts transparent

        # Convert back to an image and save
        new_image = Image.fromarray(new_img_data)
        color_name = "_".join(map(str, color))  # Name the image by the color value
        new_image.save(os.path.join(output_folder, f"{base_name}_color_{color_name}.png"))
        print(f"Saved: {base_name}_color_{color_name}.png")



# Function to count the unique colors in an image
def count_unique_colors(image_path):
    # Open the image and convert it to RGBA (if not already)
    img = Image.open(image_path).convert("RGBA")
    
    # Convert the image to a numpy array
    img_data = np.array(img)
    
    # Reshape the array to a 2D array of RGB values (ignoring alpha)
    rgb_data = img_data[:, :, :3].reshape(-1, 3)
    
    # Find the unique colors
    unique_colors = np.unique(rgb_data, axis=0)
    
    # Count the unique colors
    unique_color_count = len(unique_colors)
    
    print(f"Number of unique colors: {unique_color_count}")
    return unique_colors
