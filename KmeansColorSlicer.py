import gradio as gr
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import shutil

def kmeans_discretize(image_path, n_colors, key_colors=None):
    # Load the image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    img_data = np.array(img)
    reshaped_data = img_data.reshape(-1, 3)

    if key_colors:
        key_colors_rgb = [tuple(map(int, color.strip("()").split(","))) for color in key_colors]
        remaining_clusters = n_colors - len(key_colors_rgb)

        if remaining_clusters > 0:
            kmeans = KMeans(n_clusters=remaining_clusters, random_state=0, n_init="auto")
            kmeans.fit(reshaped_data)
            additional_centers = kmeans.cluster_centers_
            centers = np.vstack((key_colors_rgb, additional_centers))
        else:
            centers = np.array(key_colors_rgb)
    else:
        kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init="auto")
        kmeans.fit(reshaped_data)
        centers = kmeans.cluster_centers_

    labels = np.argmin(np.linalg.norm(reshaped_data[:, None] - centers, axis=2), axis=1)
    discretized_image_data = centers[labels].reshape(img_data.shape).astype(np.uint8)
    discretized_image = Image.fromarray(discretized_image_data)

    return discretized_image

def separate_colors(image, output_dir, precision=0):
    img = Image.open(image).convert("RGBA")
    img_data = np.array(img)
    rgb_data = img_data[:, :, :3].reshape(-1, 3)

    # Round the colors if precision > 0
    if precision > 0:
        rgb_data = np.round(rgb_data / (10 ** precision)) * (10 ** precision)

    unique_colors = np.unique(rgb_data, axis=0)

    # Create an output directory for color-separated images
    os.makedirs(output_dir, exist_ok=True)

    separated_images = []

    for idx, color in enumerate(unique_colors):
        mask = np.all(rgb_data == color, axis=1)
        mask_reshaped = mask.reshape(img_data[:, :, 0].shape)

        # Create an image with only the specific color visible
        new_img_data = np.zeros_like(img_data)
        new_img_data[mask_reshaped] = img_data[mask_reshaped]

        # Make other parts transparent
        new_img_data[~mask_reshaped, 3] = 0

        new_image = Image.fromarray(new_img_data)
        color_name = "_".join(map(str, color.astype(int)))
        output_path = os.path.join(output_dir, f"color_{color_name}.png")
        new_image.save(output_path)
        separated_images.append(output_path)

    return separated_images

def process_image(input_image, n_colors, key_colors):
    # Create temporary directories
    temp_dir = "temp_output"
    os.makedirs(temp_dir, exist_ok=True)

    # Apply KMeans discretization
    discretized_image = kmeans_discretize(input_image, n_colors, key_colors)
    discretized_path = os.path.join(temp_dir, "discretized_image.png")
    discretized_image.save(discretized_path)

    # Separate colors from the discretized image
    color_layers = separate_colors(discretized_path, temp_dir)

    # Zip all output images
    zip_path = os.path.join(temp_dir, "color_layers.zip")
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', temp_dir)

    return zip_path

# Define Gradio interface
def pipeline(image, n_colors, key_colors):
    if key_colors:
        key_colors = key_colors.split(";")
    output_zip = process_image(image, n_colors, key_colors)
    return output_zip

interface = gr.Interface(
    fn=pipeline,  # Connects the pipeline function
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Slider(2, 20, step=1, value=5, label="Number of Colors"),
        gr.Textbox(label="Key Colors (comma-separated RGB values, e.g., '(255,0,0);(0,255,0)')", placeholder="Optional")
    ],
    outputs=gr.File(label="Download Color Layers")
)

if __name__ == "__main__":
    interface.launch(share=True)