import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from absl import app
import os
from tqdm.auto import tqdm # Optional: for progress bar


# Define constants for clarity
STAGE_GRAYSCALE_LOW = 'grayscale_low' # Input to ColTran Core (e.g., 64x64x1)
STAGE_COARSE_COLOR = 'coarse_color'   # Output of ColTran Core (sampled, 64x64x1, indices 0-511)
STAGE_COLOR_UPSAMPLED = 'color_upsampled' # Output of Color Upsampler (e.g., 64x64x3, RGB 0-255)
STAGE_SPATIAL_UPSAMPLED = 'spatial_upsampled' # Final Output (e.g., 256x256x3, RGB 0-255)

def map_coarse_color_to_rgb(indices_tensor):
    """
    Maps a tensor of coarse color indices (0-511) to an RGB image tensor.
    Assumes indices map to an 8x8x8 RGB cube (3 bits per channel).
    """
    # Ensure input is int32
    indices = tf.cast(indices_tensor, tf.int32)

    # Calculate 3-bit R, G, B values (assuming R is most significant)
    r_3bit = indices // 64         # 0-7
    g_3bit = (indices // 8) % 8    # 0-7
    b_3bit = indices % 8           # 0-7

    # Scale 3-bit values (0-7) to 8-bit values (0-255)
    scale_factor = 255.0 / 7.0
    r_8bit = tf.cast(tf.round(tf.cast(r_3bit, tf.float32) * scale_factor), tf.uint8)
    g_8bit = tf.cast(tf.round(tf.cast(g_3bit, tf.float32) * scale_factor), tf.uint8)
    b_8bit = tf.cast(tf.round(tf.cast(b_3bit, tf.float32) * scale_factor), tf.uint8)

    # Stack channels to create RGB image
    rgb_image = tf.stack([r_8bit, g_8bit, b_8bit], axis=-1)
    return rgb_image

def visualize_coltran_stage(image_tensor, stage, title=None):
    """
    Visualizes the image tensor based on the specified ColTran stage.

    Args:
        image_tensor (tf.Tensor): The TensorFlow tensor containing image data.
                                   Expected shapes vary based on the stage.
        stage (str): The stage of the ColTran process this tensor represents.
                     Use constants like STAGE_GRAYSCALE_LOW, STAGE_COARSE_COLOR, etc.
        title (str, optional): Custom title for the plot. Defaults to stage name.
    """
    if not isinstance(image_tensor, tf.Tensor):
        raise TypeError("image_tensor must be a tf.Tensor")

    # Remove batch dimension (assuming batch size is 1)
    if image_tensor.shape[0] != 1:
        print(f"Warning: Batch size is {image_tensor.shape[0]}, visualizing only the first item.")
    vis_tensor = image_tensor[0]

    plt.figure()
    plot_title = title if title else stage.replace('_', ' ').title()
    plt.title(plot_title)
    plt.axis('off')

    try:
        if stage == STAGE_GRAYSCALE_LOW:
            # Expects shape (H, W, 1), dtype int32 or float32
            if vis_tensor.shape[-1] != 1:
                raise ValueError(f"Expected 1 channel for grayscale, got {vis_tensor.shape[-1]}")
            # Squeeze the channel dimension and convert to uint8 for display
            gray_image = tf.squeeze(vis_tensor, axis=-1)
            # Assuming the int32 values are already scaled appropriately (0-255)
            # If they are floats (e.g., 0-1), adjust casting/scaling
            gray_image = tf.cast(tf.clip_by_value(gray_image, 0, 255), tf.uint8)
            plt.imshow(gray_image.numpy(), cmap='gray', vmin=0, vmax=255)

        elif stage == STAGE_COARSE_COLOR:
            # Expects shape (H, W, 1), dtype int32 (indices 0-511)
            if vis_tensor.shape[-1] != 1:
                 raise ValueError(f"Expected 1 channel for coarse color indices, got {vis_tensor.shape[-1]}")
            # Map indices to RGB
            rgb_image = map_coarse_color_to_rgb(tf.squeeze(vis_tensor, axis=-1))
            plt.imshow(rgb_image.numpy())

        elif stage == STAGE_COLOR_UPSAMPLED or stage == STAGE_SPATIAL_UPSAMPLED:
            # Expects shape (H, W, 3), dtype int32 or uint8 (RGB 0-255)
            if vis_tensor.shape[-1] != 3:
                 raise ValueError(f"Expected 3 channels for RGB, got {vis_tensor.shape[-1]}")
            # Cast to uint8 for display
            rgb_image = tf.cast(tf.clip_by_value(vis_tensor, 0, 255), tf.uint8)
            plt.imshow(rgb_image.numpy())

        else:
            raise ValueError(f"Unknown stage: {stage}. Valid stages are: "
                             f"{STAGE_GRAYSCALE_LOW}, {STAGE_COARSE_COLOR}, "
                             f"{STAGE_COLOR_UPSAMPLED}, {STAGE_SPATIAL_UPSAMPLED}")

        plt.show()

    except Exception as e:
        print(f"Error visualizing stage '{stage}': {e}")
        print(f"Input tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
        plt.close() # Close the potentially empty figure on error

# --- Example Usage ---

# 1. Example Grayscale Input (like your gray_64)
#    Simulating the input tensor

def visualize_logits_confidence(logits_tensor, title="Prediction Confidence (Max Logit)", cmap='viridis'):
    """Visualizes the most likely color based on logits."""
    if not isinstance(logits_tensor, tf.Tensor):
        raise TypeError("logits_tensor must be a tf.Tensor")
    if len(logits_tensor.shape) != 4 or logits_tensor.shape[-1] != 512:
         raise ValueError(f"Expected shape (1, H, W, 512), got {logits_tensor.shape}")
    if logits_tensor.shape[0] != 1:
        print(f"Warning: Batch size is {logits_tensor.shape[0]}, visualizing only the first item.")

    # Get the index (0-511) of the highest logit for each pixel
    # Argmax across the last dimension (the 512 channels)
    predicted_indices = tf.argmax(logits_tensor[0], axis=-1) # Shape becomes (64, 64)

    # Map indices to RGB colors
    rgb_image = map_coarse_color_to_rgb(predicted_indices)

    plt.figure()
    plt.imshow(rgb_image.numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def visualize_parallel_prediction(indices_tensor, title="Parallel Head Predicted Coarse Color"):
    """Visualizes the coarse color indices predicted by the parallel head."""
    if not isinstance(indices_tensor, tf.Tensor):
        raise TypeError("indices_tensor must be a tf.Tensor")
    if len(indices_tensor.shape) != 3: # Expecting (Batch, H, W)
         raise ValueError(f"Expected shape (1, H, W), got {indices_tensor.shape}")
    if indices_tensor.shape[0] != 1:
        print(f"Warning: Batch size is {indices_tensor.shape[0]}, visualizing only the first item.")

    # Indices are already computed (no argmax needed here)
    predicted_indices = indices_tensor[0] # Shape becomes (64, 64)

    # Map indices to RGB colors
    rgb_image = map_coarse_color_to_rgb(predicted_indices)

    plt.figure()
    plt.imshow(rgb_image.numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def normalize_channel(channel_data):
    """Normalizes a single channel tensor (H, W) to 0-1 float range."""
    min_val = tf.reduce_min(channel_data)
    max_val = tf.reduce_max(channel_data)
    if max_val == min_val:
        # Handle uniform channel - return a mid-value or zero tensor
        # Returning zeros avoids division by zero and results in a black image
        return tf.zeros_like(channel_data, dtype=tf.float32)
    else:
        # Normalize to 0-1 range
        normalized = (channel_data - min_val) / (max_val - min_val)
        return normalized

def export_logits_channels(logits_tensor, output_dir, prefix="logit_channel", use_matplotlib=True):
    """
    Exports each channel of a logit tensor as a normalized grayscale image.

    Args:
        logits_tensor (tf.Tensor): The input tensor with shape (1, H, W, C)
                                    (e.g., (1, 64, 64, 512)) and dtype float32.
        output_dir (str): The directory where the images will be saved.
        prefix (str, optional): A prefix for the output filenames.
                                Defaults to "logit_channel".
        use_matplotlib (bool, optional): If True, use matplotlib.pyplot.imsave.
                                         If False, use PIL (requires Pillow installed).
                                         Defaults to True.
    """
    # --- Input Validation ---
    if not isinstance(logits_tensor, tf.Tensor):
        raise TypeError("logits_tensor must be a tf.Tensor")
    if len(logits_tensor.shape) != 4:
        raise ValueError(f"Expected 4 dimensions (1, H, W, C), got {len(logits_tensor.shape)}")
    if logits_tensor.shape[0] != 1:
        print(f"Warning: Batch size is {logits_tensor.shape[0]}, processing only the first item.")
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    if not use_matplotlib:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required when use_matplotlib=False. Please install it: pip install Pillow")

    # --- Processing ---
    batch_slice = logits_tensor[0] # Remove batch dimension -> (H, W, C)
    num_channels = batch_slice.shape[-1]
    height, width = batch_slice.shape[0], batch_slice.shape[1]

    print(f"Exporting {num_channels} channels ({height}x{width}) to {output_dir}...")

    for i in tqdm(range(num_channels)): # Use tqdm for progress bar
        # 1. Extract the channel
        channel_data = batch_slice[..., i] # Shape (H, W)

        # 2. Normalize the channel to 0-1
        normalized_data = normalize_channel(channel_data)

        # 3. Scale to 0-255 and convert to uint8
        image_data_uint8 = tf.cast(normalized_data * 255.0, tf.uint8).numpy()

        # 4. Construct filename (zero-padded index)
        filename = os.path.join(output_dir, f"{prefix}_{i:03d}.png")

        # 5. Save the image
        try:
            if use_matplotlib:
                plt.imsave(filename, image_data_uint8, cmap='gray', vmin=0, vmax=255)
            else:
                 # Use Pillow
                 img = Image.fromarray(image_data_uint8, mode='L') # 'L' mode for grayscale
                 img.save(filename)
        except Exception as e:
            print(f"Error saving channel {i} to {filename}: {e}")

    print(f"Finished exporting {num_channels} images.")
    
def normalize_channel(channel_data):
    """Normalizes a single channel tensor (H, W) to 0-1 float range."""
    min_val = tf.reduce_min(channel_data)
    max_val = tf.reduce_max(channel_data)
    if max_val == min_val:
        return tf.zeros_like(channel_data, dtype=tf.float32)
    else:
        normalized = (channel_data - min_val) / (max_val - min_val)
        return normalized

def export_logits_colormaps(logits_tensor, output_dir, prefix="logit_cmap", cmap='viridis'):
    """
    Exports each channel of a logit tensor as a normalized RGB heatmap image
    using a specified colormap. Saves 512 images.

    Args:
        logits_tensor (tf.Tensor): The input tensor with shape (1, H, W, C)
                                    (e.g., (1, 64, 64, 512)) and dtype float32.
        output_dir (str): The directory where the images will be saved.
        prefix (str, optional): A prefix for the output filenames.
                                Defaults to "logit_cmap".
        cmap (str, optional): The matplotlib colormap to use (e.g., 'viridis',
                              'jet', 'gray', 'magma'). Defaults to 'viridis'.
    """
    # --- Input Validation ---
    if not isinstance(logits_tensor, tf.Tensor):
        raise TypeError("logits_tensor must be a tf.Tensor")
    if len(logits_tensor.shape) != 4:
         raise ValueError(f"Expected 4 dimensions (1, H, W, C), got {len(logits_tensor.shape)}")
    if logits_tensor.shape[0] != 1:
        print(f"Warning: Batch size is {logits_tensor.shape[0]}, processing only the first item.")
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    try:
        plt.get_cmap(cmap) # Check if cmap is valid
    except ValueError:
        print(f"Warning: Colormap '{cmap}' not found. Using 'viridis'.")
        cmap = 'viridis'


    # --- Processing ---
    batch_slice = logits_tensor[0] # Remove batch dimension -> (H, W, C)
    num_channels = batch_slice.shape[-1]
    height, width = batch_slice.shape[0], batch_slice.shape[1]

    print(f"Exporting {num_channels} colormapped channels ({height}x{width}) to {output_dir} using cmap='{cmap}'...")

    for i in tqdm(range(num_channels)): # Use tqdm for progress bar
        # 1. Extract the channel
        channel_data = batch_slice[..., i] # Shape (H, W)

        # 2. Normalize the channel to 0-1 (important for consistent colormapping)
        normalized_data = normalize_channel(channel_data).numpy() # Convert to numpy for imsave

        # 3. Construct filename
        filename = os.path.join(output_dir, f"{prefix}_{cmap}_{i:03d}.png")

        # 4. Save the image using the colormap
        # plt.imsave handles applying the colormap to the normalized data
        try:
            plt.imsave(filename, normalized_data, cmap=cmap)
        except Exception as e:
            print(f"Error saving channel {i} colormap to {filename}: {e}")

    print(f"Finished exporting {num_channels} colormapped images.")

def map_coarse_color_to_rgb(indices_tensor):
    """Maps a tensor of coarse color indices (0-511) to an RGB image tensor."""
    indices = tf.cast(indices_tensor, tf.int32)
    r_3bit = indices // 64         # 0-7
    g_3bit = (indices // 8) % 8    # 0-7
    b_3bit = indices % 8           # 0-7
    scale_factor = 255.0 / 7.0
    r_8bit = tf.cast(tf.round(tf.cast(r_3bit, tf.float32) * scale_factor), tf.uint8)
    g_8bit = tf.cast(tf.round(tf.cast(g_3bit, tf.float32) * scale_factor), tf.uint8)
    b_8bit = tf.cast(tf.round(tf.cast(b_3bit, tf.float32) * scale_factor), tf.uint8)
    rgb_image = tf.stack([r_8bit, g_8bit, b_8bit], axis=-1)
    return rgb_image

def save_logits_derived_rgb(logits_tensor, output_dir, prefix="predicted_rgb"):
    """
    Calculates the most likely coarse color for each pixel based on logits,
    maps it to RGB, and saves it as a single RGB image file using the
    specified output directory and prefix.

    Args:
        logits_tensor (tf.Tensor): Input tensor shape (1, H, W, 512), dtype float32.
        output_dir (str): The directory where the image will be saved.
        prefix (str, optional): A prefix for the output filename.
                                Defaults to "predicted_rgb".
                                The final filename will be <output_dir>/<prefix>.png.
    """
    # --- Input Validation ---
    if not isinstance(logits_tensor, tf.Tensor):
        raise TypeError("logits_tensor must be a tf.Tensor")
    if len(logits_tensor.shape) != 4 or logits_tensor.shape[-1] != 512:
         raise ValueError(f"Expected shape (1, H, W, 512), got {logits_tensor.shape}")
    if logits_tensor.shape[0] != 1:
        print(f"Warning: Batch size is {logits_tensor.shape[0]}, processing only the first item.")

    # --- Ensure Output Directory Exists ---
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # --- Construct Filename ---
    # Ensure the filename ends with a standard image extension
    if not prefix.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
         filename_base = prefix
         extension = ".png" # Default to PNG
    else:
        filename_base = os.path.splitext(prefix)[0]
        extension = os.path.splitext(prefix)[1]

    # Construct the full path
    output_filename = os.path.join(output_dir, f"{filename_base}{extension}")


    # --- Processing ---
    # 1. Find the Most Likely Coarse Color Index
    predicted_indices = tf.argmax(tf.stop_gradient(logits_tensor[0]), axis=-1) # Shape -> (H, W)

    # 2. Map Index to RGB
    rgb_image_tensor = map_coarse_color_to_rgb(predicted_indices) # Shape -> (H, W, 3)

    # 3. Convert to NumPy array for saving
    rgb_image_numpy = rgb_image_tensor.numpy()

    # 4. Save using matplotlib.pyplot.imsave
    try:
        plt.imsave(output_filename, rgb_image_numpy)
        print(f"Successfully saved predicted RGB image to: {output_filename}")
    except Exception as e:
        print(f"Error saving image to {output_filename}: {e}")
    
output_directory = "./logit_channels_output"


def main(_):
    example_gray_low = tf.constant(np.random.randint(0, 256, size=(1, 64, 64, 1)), dtype=tf.int32)
    visualize_coltran_stage(example_gray_low, STAGE_GRAYSCALE_LOW)

    # 2. Example Coarse Color Output (Simulated)
    #    Random indices between 0 and 511
    example_coarse_indices = tf.constant(np.random.randint(0, 512, size=(1, 64, 64, 1)), dtype=tf.int32)
    visualize_coltran_stage(example_coarse_indices, STAGE_COARSE_COLOR)

    # 3. Example Color Upsampled Output (Simulated)
    #    Random RGB values
    example_color_upsampled = tf.constant(np.random.randint(0, 256, size=(1, 64, 64, 3)), dtype=tf.int32) # Or uint8
    visualize_coltran_stage(example_color_upsampled, STAGE_COLOR_UPSAMPLED)

    # 4. Example Spatial Upsampled Output (Simulated)
    #    Random RGB values at higher res
    example_spatial_upsampled = tf.constant(np.random.randint(0, 256, size=(1, 256, 256, 3)), dtype=tf.int32) # Or uint8
    visualize_coltran_stage(example_spatial_upsampled, STAGE_SPATIAL_UPSAMPLED)

if __name__ == '__main__':
  app.run(main)
