import pandas as pd
from sklearn.model_selection import train_test_split

from absl import flags
from absl import app

flags.DEFINE_string('csv_dir', None, 'Csv directory for comment of images.')
flags.DEFINE_string('save_dir', None, 'save directory for split csv')
FLAGS = flags.FLAGS

def split_dataset(csv_path, save_path):
    # 1️⃣ Load your CSV
    df = pd.read_csv(csv_path, sep="|")  # or ',' if your CSV uses commas
    df.columns = df.columns.str.strip()
    print(df.head())

    # 2️⃣ Split train (80%) / temp (20%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # 3️⃣ Split temp into validation (10%) and test (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,shuffle=True)

    # 4️⃣ Save them
    train_df.to_csv(save_path+"/flickr30k_train.csv", index=False,sep='|')
    val_df.to_csv(save_path+"/flickr30k_val.csv", index=False,sep='|')
    test_df.to_csv(save_path+"/flickr30k_test.csv", index=False,sep='|')

    print("✅ Done! Splits saved:")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

def main(_):
    csv_path = FLAGS.csv_dir
    save_path=FLAGS.save_dir
    split_dataset(csv_path,save_path)


if __name__ == '__main__':
  app.run(main)
