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



def preprocess_flickr30k_captions(csv_path,batch_size=64,output_path="./coltran/dataset"):
    """
    Reads a Flickr30k CSV, encodes all captions using CLIP, and saves them to a .npz file.

    Args:
        csv_path (str): Path to the Flickr30k captions CSV file.
        output_path (str): Path to save the output .npz file.
        batch_size (int): Number of captions to process at once for efficiency.
    """
    print("Loading captions from:", csv_path)
    # Adjust column names based on your actual CSV file
    df = pd.read_csv(csv_path, sep="|")
    df.columns = df.columns.str.strip()
    df["comment"] = df["comment"].fillna("").astype(str)

    # Let's assume the columns are 'image_name' and 'caption_text'
    # It's common for Flickr30k to have 5 captions per image. We will encode all of them.
    captions = df['comment'].tolist()
    image_names = df['image_name'].tolist()

    print(f"Found {len(captions)} total captions to encode.")

    text_encoder = TextEncoder()
    all_embeddings = []

    print("Starting batch encoding...")
    # Process in batches for efficiency
    for i in tqdm(range(0, len(captions), batch_size)):
        batch_texts = captions[i:i + batch_size]
        batch_embeddings = text_encoder.encode_batch(batch_texts)
        all_embeddings.append(batch_embeddings)

    # Concatenate all batch results into one large NumPy array
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    print("Encoding complete.")
    print("Shape of final embeddings array:", final_embeddings.shape)

    # Save the results. A .npz file is like a zip for NumPy arrays.
    # It's a good way to store multiple related arrays.
    np.savez_compressed(
        output_path,
        image_names=np.array(image_names),
        captions=np.array(captions),
        embeddings=final_embeddings
    )
    print(f"Successfully saved embeddings to: {output_path}")

def main(_):
    csv_path = FLAGS.csv_dir
    output_dir=FLAGS.output_dir
    batch_size = FLAGS.batch_size
    preprocess_flickr30k_captions(csv_path=csv_path,output_path=output_dir,batch_size=batch_size)

if __name__ == '__main__':
  app.run(main)