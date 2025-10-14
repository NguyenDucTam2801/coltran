# import tensorflow as tf
import tensorflow.compat.v2 as tf
import argparse
import os

def read_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: {checkpoint_dir} is not a valid directory.")
        return

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint:
        print(f"Latest checkpoint found: {checkpoint}")
        reader = tf.train.load_checkpoint(checkpoint)
        print("Checkpoint variables:")
        for key in reader.get_variable_to_shape_map():
            print(f"Variable: {key}, Shape: {reader.get_variable_to_shape_map()[key]}")
    else:
        print("No checkpoint found in the directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read TensorFlow checkpoint.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory path where the checkpoint file is saved."
    )
    args = parser.parse_args()
    read_checkpoint(args.checkpoint_dir)