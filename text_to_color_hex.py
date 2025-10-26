import spacy
from spacy.matcher import Matcher
import matplotlib.colors as mcolors
import tensorflow as tf

# --- Setup: Do this once ---

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Create a comprehensive list of color names
# We'll combine CSS colors and the much larger XKCD color list
color_names = set(mcolors.CSS4_COLORS.keys())
color_names.update(mcolors.XKCD_COLORS.keys())

# Remove single-letter names that can cause false positives (like 'k' for black)
color_names = {name for name in color_names if len(name) > 1}

# Create matcher patterns. We want to match case-insensitively.
# The pattern is a list of dictionaries, one for each token.
# We will match single-word colors like "red" and "blue".
patterns = [[{'LOWER': name}] for name in color_names]

# Initialize the Matcher with the spaCy vocabulary
matcher = Matcher(nlp.vocab)
matcher.add("COLOR_PATTERN", patterns)


# --- The Function to Use ---

def filter_color_words(caption):
    """
    Filters and returns color words from a caption using a pre-defined list.

    Args:
        caption (str): The input sentence.

    Returns:
        list: A list of unique color words found in the caption.
    """
    # Process the caption with spaCy
    doc = nlp(caption)

    # Find all matches in the document
    matches = matcher(doc)

    # Extract the matched text (the color word)
    # Use a set to store unique color names found
    found_colors = set()
    for match_id, start, end in matches:
        color_span = doc[start:end]
        found_colors.add(color_span.text.lower())
    color_list=list(found_colors)
    hex_palette = []
    for name in color_list:
        try:
            # mcolors.to_hex() is the direct function for this!
            # It can handle names like 'red', 'skyblue', 'darkgreen'
            # For multi-word names, we replace spaces.
            hex_code = mcolors.to_hex(name.replace(" ", ""))
            hex_palette.append(hex_code)
        except ValueError:
            # Handle cases where the color name is not recognized
            print(f"Warning: Color name '{name}' not found. Skipping.")
            hex_palette.append(None)
    return hex_palette


def get_coarse_color_palette_tf(n_bits=3):
    """
    Generates the ColTran 3-bit (512 colors) RGB palette as a TensorFlow tensor.
    """
    num_values = 2 ** n_bits
    channel_values = tf.linspace(0.0, 255.0, num_values)
    channel_values = tf.cast(tf.round(channel_values), dtype=tf.float32)

    R, G, B = tf.meshgrid(channel_values, channel_values, channel_values, indexing='ij')
    palette_rgb = tf.stack([R, G, B], axis=-1)
    return tf.reshape(palette_rgb, [-1, 3])  # Shape: (512, 3)


def find_closest_coarse_color_indices(hex_colors, coarse_palette):
    """
    Finds the indices (0-511) in the coarse palette that are closest to
    the given hex colors.

    Args:
        hex_colors (list): A list of hex color strings (e.g., ["#008000"]).
        coarse_palette (tf.Tensor): The (512, 3) coarse RGB palette.

    Returns:
        tf.Tensor: A tensor of the closest indices.
    """
    # Convert hex colors to a tensor of RGB values
    rgb_colors = [mcolors.to_rgb(h) for h in hex_colors]
    target_rgb_tensor = tf.constant(rgb_colors, dtype=tf.float32) * 255.0  # Shape: (num_colors, 3)

    # Use broadcasting to calculate squared Euclidean distance
    # Reshape for broadcasting:
    #   - palette:  (1, 512, 3)
    #   - target:   (num_colors, 1, 3)
    #   - diff:     (num_colors, 512, 3)
    diff = tf.expand_dims(coarse_palette, 0) - tf.expand_dims(target_rgb_tensor, 1)
    distances = tf.reduce_sum(tf.square(diff), axis=-1)  # Shape: (num_colors, 512)

    # Find the index of the minimum distance for each target color
    closest_indices = tf.argmin(distances, axis=1, output_type=tf.int32)

    return closest_indices


# --- Step 3, 4, & 5: The main function ---

def create_masked_coarse_color_map(size=(64, 64), masks=None, color_indices=None):
    """
    Creates a coarse color label map (0-511) where specified regions are
    filled with given colors, and the rest is random.

    Args:
        size (tuple): The (height, width) of the output map.
        masks (list of np.ndarray): A list of boolean masks. Each mask should have
                                    the shape `size`.
        color_indices (list of int): A list of coarse color indices (0-511)
                                     corresponding to each mask.

    Returns:
        tf.Tensor: A tensor of shape `size` with the final coarse color labels.
    """
    height, width = size

    # Start with a map of random integers from 0 to 511
    final_map = tf.random.uniform(shape=size, minval=0, maxval=512, dtype=tf.int32)

    if masks is None or color_indices is None:
        return final_map  # Return the purely random map if no masks are provided

    if len(masks) != len(color_indices):
        raise ValueError("The number of masks must match the number of color indices.")

    # Iterate through the masks and apply the colors from highest priority to lowest
    # (If masks overlap, the last one in the list will win)
    for mask, color_idx in zip(masks, color_indices):
        mask_tensor = tf.constant(mask, dtype=tf.bool)

        # Create a tensor of the desired color, matching the map's shape
        color_fill = tf.fill(dims=size, value=color_idx)

        # Use tf.where to apply the color only in the masked region
        final_map = tf.where(mask_tensor, color_fill, final_map)

    return final_map