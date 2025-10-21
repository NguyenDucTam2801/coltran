import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


# Function to parse the TFRecord file
def parse_tfrecord_fn(example_proto):
    feature_description = {
        'image': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

output_dir="./checkpoint/result/"
tfrecord_file = os.path.join(output_dir, 'gen0.tfrecords')
# Reading the TFRecord file
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
print(parsed_dataset)

for example in parsed_dataset.take(1):
    print("Label:", example["label"].numpy())
    print("Image (flattened):", example["image"].numpy().shape, "...")
    img_flat=example["image"]
    img_rgb = tf.reshape(img_flat, (64, 64, 3))
    img_rgb = tf.cast(img_rgb, tf.uint8)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()