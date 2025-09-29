import pandas as pd
from tqdm.auto import tqdm
import numpy as np

from text_encode import TextEncoder

from absl import flags
from absl import app


flags.DEFINE_string('csv_dir', None, 'Csv directory for comment of images.')
flags.DEFINE_string('output_dir', None, 'output dir for embedding comments.')
flags.DEFINE_integer('batch_size', 64, 'batch size for encoding comments.')
FLAGS = flags.FLAGS


def preprocess_flickr30k_captions(csv_path, output_path, batch_size=64):
    """
    Reads a Flickr30k CSV, encodes all captions using CLIP, and saves both
    pooled and sequential embeddings to a .npz file.
    """
    print("Loading captions from:", csv_path)
    df = pd.read_csv(csv_path,sep="|")
    df.columns = df.columns.str.strip()

    initial_rows = len(df)
    df.dropna(subset=['comment'], inplace=True)
    if len(df) < initial_rows:
        print(f"Warning: Removed {initial_rows - len(df)} rows with missing captions.")

    captions = df['comment'].tolist()
    image_names = df['image_name'].tolist()

    print(f"Found {len(captions)} total captions to encode.")

    text_encoder = TextEncoder()
    all_pooled_embeddings = []
    all_hidden_state_embeddings = []

    print("Starting batch encoding...")
    for i in tqdm(range(0, len(captions), batch_size)):
        batch_texts = captions[i:i + batch_size]
        # print(f"Encoding {batch_texts} captions...")
        try:
            pooled_embeds = text_encoder.encode_batch(batch_texts)
        except Exception as e:
            print(f"Error batch_textsL {batch_texts}")
            print(f"An error occurred during encoding: {e}")

        all_pooled_embeddings.append(pooled_embeds)

    # Concatenate all batch results into two large NumPy arrays
    final_pooled = np.concatenate(all_pooled_embeddings, axis=0)

    print("Encoding complete.")
    print("Shape of final pooled embeddings array:", final_pooled.shape)

    # Save both embedding types in the same file
    np.savez_compressed(
        output_path,
        image_names=np.array(image_names),
        # Save pooled_output for global conditioning experiments
        pooled_embeddings=final_pooled,
        # Save last_hidden_state for cross-attention experiments
    )
    print(f"Successfully saved both embedding types to: {output_path}")


def main(_):
    csv_path = FLAGS.csv_dir
    output_dir=FLAGS.output_dir
    batch_size = FLAGS.batch_size
    preprocess_flickr30k_captions(csv_path=csv_path,output_path=output_dir,batch_size=batch_size)

if __name__ == '__main__':
  app.run(main)
