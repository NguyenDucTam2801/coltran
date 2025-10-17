import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import spacy


from text_encode import TextEncoder

from absl import flags
from absl import app


flags.DEFINE_string('csv_dir', None, 'Csv directory for comment of images.')
flags.DEFINE_string('output_dir', "/embedding/embedded.npz", 'output dir for embedding comments.')
flags.DEFINE_integer('batch_size', 64, 'batch size for encoding comments.')
FLAGS = flags.FLAGS


def preprocess_flickr30k_captions(csv_path, output_path, batch_size=64):
    """
    Reads a Flickr30k CSV, encodes all captions using CLIP, and saves both
    pooled and sequential embeddings to a .npz file.
    """
    nlp = spacy.load("en_core_web_sm")
    '''
    Remember update spacy, typer, click, numpy
    $ pip install -U spacy typer click numpy

    '''
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
    all_pooled_embeddings_caption=[]
    all_pooled_embeddings_adjective = []

    print("Starting batch encoding...")
    for i in tqdm(range(0, len(captions), batch_size)):
        batch_texts = captions[i:i+batch_size]
        doc=nlp.pipe(batch_texts)
        adjectives = []
        for token in doc:
            adjective_temp = []
            for t in token:
                if t.pos_ == "ADJ":
                    adjective_temp.append(t.text.lower())
            # print(f"adjective_temp:{adjective_temp}")
            adjectives.append((", ".join(adjective_temp) if len(adjective_temp)>0 else "colored")+".")

        pooled_embeds_caption = text_encoder.encode_batch(batch_texts)
        pooled_embeds_adjective = text_encoder.encode_batch(adjectives)
        # print(f"pooled_embeds_adjective:{pooled_embeds_adjective.shape}")

        all_pooled_embeddings_caption.append(pooled_embeds_caption)
        all_pooled_embeddings_adjective.append(pooled_embeds_adjective)




    # Concatenate all batch results into two large NumPy arrays
    final_pooled_caption = np.concatenate(all_pooled_embeddings_caption,axis=0)
    final_pooled_adjective = np.concatenate(all_pooled_embeddings_adjective, axis=0)


    print("Encoding complete.")
    print("Shape of final pooled embeddings captions array:", final_pooled_caption.shape)
    print("Shape of final pooled embeddings adjectives array:", final_pooled_adjective.shape)

    # Save both embedding types in the same file
    np.savez_compressed(
        output_path,
        image_names=np.array(image_names),
        # Save pooled_output for global conditioning experiments
        embeddings_caption=final_pooled_caption,
        embeddings_adj=final_pooled_adjective,


    )
    print(f"Successfully saved both embedding types to: {output_path}")


def main(_):
    csv_path = FLAGS.csv_dir
    output_dir=FLAGS.output_dir
    batch_size = FLAGS.batch_size
    preprocess_flickr30k_captions(csv_path=csv_path,output_path=output_dir,batch_size=batch_size)

if __name__ == '__main__':
  app.run(main)
