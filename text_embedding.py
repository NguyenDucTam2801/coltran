import pandas as pd
import h5py
from tqdm.auto import tqdm
import numpy as np
import os
import gc # Garbage Collector

from text_encode import TextEncoder

from absl import flags
from absl import app


flags.DEFINE_string('csv_dir', None, 'Csv directory for comment of images.')
flags.DEFINE_string('output_dir', None, 'output dir for embedding comments.')
flags.DEFINE_integer('batch_size', 64, 'batch size for encoding comments.')
FLAGS = flags.FLAGS


def csv_to_hdf5_flickr30k(csv_path, output_hdf5_path, batch_size=64):
    """
    Reads a Flickr30k CSV, encodes all captions using CLIP, and saves both
    pooled and sequential embeddings directly to an HDF5 file.

    Args:
        csv_path (str): Path to the Flickr30k CSV file
        output_hdf5_path (str): Path for the final HDF5 file to be created
        batch_size (int): Batch size for encoding (default: 64)
    """
    print("Loading captions from:", csv_path)

    # Read and preprocess CSV
    df = pd.read_csv(csv_path, sep="|")
    df.columns = df.columns.str.strip()

    initial_rows = len(df)
    df.dropna(subset=['comment'], inplace=True)
    if len(df) < initial_rows:
        print(f"Warning: Removed {initial_rows - len(df)} rows with missing captions.")

    captions = df['comment'].tolist()
    image_names = df['image_name'].tolist()

    print(f"Found {len(captions)} total captions to encode.")

    # Initialize text encoder
    text_encoder = TextEncoder()

    # Get sample embedding to determine shapes
    sample_embed_pooled, sample_embed_hidden = text_encoder.encode_batch([captions[0]])

    # Determine final shapes
    pooled_shape = (len(captions),) + sample_embed_pooled.shape[1:]
    hidden_shape = (len(captions),) + sample_embed_hidden.shape[1:]

    print(f"Final shape for 'pooled_embeddings': {pooled_shape}")
    print(f"Final shape for 'sequence_embeddings': {hidden_shape}")

    # Create HDF5 file and datasets
    print(f"Creating HDF5 file at: {output_hdf5_path}")
    with h5py.File(output_hdf5_path, 'w') as hf:
        # Create datasets with pre-allocated space
        dset_pooled = hf.create_dataset('pooled_embeddings',
                                        shape=pooled_shape,
                                        dtype=sample_embed_pooled.dtype)
        dset_hidden = hf.create_dataset('sequence_embeddings',
                                        shape=hidden_shape,
                                        dtype=sample_embed_hidden.dtype)

        # Save metadata
        hf.create_dataset('image', data=image_names, dtype=h5py.special_dtype(vlen=str))
        hf.create_dataset('caption', data=captions, dtype=h5py.special_dtype(vlen=str))

        # Process in batches
        num_batches = (len(captions) + batch_size - 1) // batch_size
        print("Starting batch encoding...")

        for i in tqdm(range(num_batches), desc="Processing Batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(captions))

            batch_texts = captions[start_idx:end_idx]

            # Get both embedding types
            pooled_embeds, hidden_state_embeds = text_encoder.encode_batch(batch_texts)

            # Write directly to HDF5 datasets
            dset_pooled[start_idx:end_idx] = pooled_embeds
            dset_hidden[start_idx:end_idx] = hidden_state_embeds

            # Clean up memory
            del pooled_embeds, hidden_state_embeds
            if i % 10 == 0:  # Collect garbage every 10 batches
                gc.collect()

    print(f"\nSuccessfully saved embeddings to: {output_hdf5_path}")
    print(f"File contains:")
    print(f"- pooled_embeddings: shape {pooled_shape}")
    print(f"- sequence_embeddings: shape {hidden_shape}")
    print(f"- image_names: {len(image_names)} items")
    print(f"- captions: {len(captions)} items")

def main(_):
    csv_path = FLAGS.csv_dir
    output_dir=FLAGS.output_dir
    batch_size = FLAGS.batch_size
    csv_to_hdf5_flickr30k(csv_path, output_dir, batch_size=batch_size)


if __name__ == '__main__':
  app.run(main)