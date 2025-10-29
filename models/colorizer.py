# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ColTran core.

Core autoregressive component of the colorization transformer based on
the AxialTransformer with conditional self-attention layers.

See Section 3 and Section 4.1 of https://openreview.net/pdf?id=5NA1PinlGFu
for more details.
"""
import tensorflow.compat.v2 as tf
# from tensorflow.compat.v2.keras import layers
from tensorflow.keras import layers
from coltran.models import core
from coltran.models import layers as coltran_layers
from coltran.utils import base_utils
from coltran import visualizer
from coltran import text_to_color_hex
import matplotlib.pyplot as plt
import uuid
import os
import spacy

tf.config.run_functions_eagerly(True)


class ColTranCore(tf.keras.Model):
  """Colorization Transformer."""

  def __init__(self, config, **kwargs):
    super(ColTranCore, self).__init__(**kwargs)
    self.config = config

    # 3 bits per channel, 8 colors per channel, a total of 512 colors.
    self.num_symbols_per_channel = 2**3
    self.num_symbols = self.num_symbols_per_channel**3
    self.gray_symbols, self.num_channels = 256, 1

    self.enc_cfg = config.encoder
    self.dec_cfg = config.decoder
    self.hidden_size = self.config.get('hidden_size',
                                       self.dec_cfg.hidden_size)

    # stage can be 'encoder_decoder' or 'decoder'
    # 1. decoder -> loss only due to autoregressive model.
    # 2. encoder_decoder -> loss due to both the autoregressive and parallel
    # model.
    # encoder_only and all
    self.stage = config.get('stage', 'decoder')
    self.is_parallel_loss = 'encoder' in self.stage
    stages = ['decoder', 'encoder_decoder']
    # Extract features from specific layers (e.g., block3_conv3)
    if self.stage not in stages:
      raise ValueError('Expected stage to be in %s, got %s' %
                       (str(stages), self.stage))

    # initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    #
    # # Cross Attention layer to condition on Grayscale encode
    # self.cross_attention_layer_grayscale_encode = tf.keras.layers.MultiHeadAttention(
    #   num_heads=8,  # A common choice, can be tuned
    #   key_dim=self.hidden_size // 8,  # The dimension of each head's Q, K, V
    #   output_shape=self.hidden_size,  # Ensure the output has the same dimension as the input
    #   name="grayscale_text_cross_attention",
    #   kernel_initializer=initializer,
    # )
    #
    # # Every attention block needs a LayerNormalization and Dropout for stability.
    # self.cross_attention_norm_grayscale_encode = tf.keras.layers.LayerNormalization()
    # self.cross_attention_dropout_grayscale_encode = tf.keras.layers.Dropout(0.1)  # Dropout rate can be tuned
    #
    # # Cross Attention layer to condition on inner decode
    # self.cross_attention_layer_inner_decode = tf.keras.layers.MultiHeadAttention(
    #   num_heads=8,  # A common choice, can be tuned
    #   key_dim=self.hidden_size // 8,  # The dimension of each head's Q, K, V
    #   output_shape=self.hidden_size,  # Ensure the output has the same dimension as the input
    #   name="grayscale_text_cross_attention",
    #   kernel_initializer=initializer,
    # )
    #
    # # Every attention block needs a LayerNormalization and Dropout for stability.
    # self.cross_attention_norm_inner_decode = tf.keras.layers.LayerNormalization()
    # self.cross_attention_dropout_inner_decode = tf.keras.layers.Dropout(0.1)  # Dropout rate can be tuned

    # self.color_scale_generator_inner = layers.Dense(self.hidden_size, activation='sigmoid', name='color_scale')
    # self.color_shift_generator_inner = layers.Dense(self.hidden_size, activation=None, name='color_shift')

  @property
  def metric_keys(self):
    if self.stage == 'encoder_decoder':
      return ['encoder']
    return []

  def build(self, input_shape):
    # encoder graph
    self.encoder = core.GrayScaleEncoder(self.enc_cfg)
    if self.is_parallel_loss:
      self.parallel_dense = layers.Dense(
          units=self.num_symbols, name='parallel_logits', use_bias=False)

    # decoder graph: outer decoder -> inner decoder -> logits.
    self.pixel_embed_layer = layers.Dense(
        units=self.hidden_size, use_bias=False)
    self.outer_decoder = core.OuterDecoder(self.dec_cfg)
    self.inner_decoder = core.InnerDecoder(self.dec_cfg)
    self.final_dense = layers.Dense(
        units=self.num_symbols, name='auto_logits')
    self.final_norm = layers.LayerNormalization()

  def call(self, inputs, embedded_captions = None ,training=True):
    # encodes grayscale (H, W) into activations of shape (H, W, 512).
    gray = tf.image.rgb_to_grayscale(inputs)
    z = self.encoder(gray, training=training)
    conditioned_z=z

    # if embedded_captions is not None:
    #
    #   captions_float = tf.cast(embedded_captions["caption"], dtype=tf.float32)
    #   batch_size, height, width, _ = tf.unstack(tf.shape(conditioned_z))
    #
    #   # 1. --- Reshape Inputs for Attention ---
    #   # Reshape image features from (B, H, W, C) to a sequence (B, H*W, C)
    #   image_seq = tf.reshape(z, [batch_size, -1, self.hidden_size])
    #
    #   # Promote the single text embedding to a sequence of length 1
    #   text_seq = tf.expand_dims(captions_float, 1)
    #
    #   # 2. --- Apply Cross-Attention (The Sub-layer) ---
    #   attention_output = self.cross_attention_layer_grayscale_encode(
    #     query=image_seq,
    #     value=text_seq,
    #     key=text_seq,
    #     training=training  # Pass the training flag
    #   )
    #
    #   # 3. --- Apply Dropout ---
    #   attention_output = self.cross_attention_dropout_grayscale_encode(attention_output, training=training)
    #
    #   # 4. --- Apply Residual Connection (The "Add") ---
    #   # This is the most important fix. Add the output back to the input.
    #   attention_result_seq = image_seq + attention_output
    #
    #   # 5. --- Apply Layer Normalization (The "Norm") ---
    #   conditioned_z_seq = self.cross_attention_norm_grayscale_encode(attention_result_seq)
    #
    #   # 6. --- Reshape back to Image Format ---
    #   conditioned_z = tf.reshape(conditioned_z_seq, [batch_size, 64, 64, self.hidden_size])

    # visualizer.export_logits_channels(logits_tensor=conditioned_z, output_dir="./logs/conditioned_z/", prefix="conditioned_z")

    if self.is_parallel_loss:
      enc_logits = self.parallel_dense(conditioned_z)
      enc_logits = tf.expand_dims(enc_logits, axis=-2)

    dec_logits = self.decoder(inputs, conditioned_z, training=training,embedded_captions=embedded_captions)
    if self.is_parallel_loss:
      return dec_logits, {'encoder_logits': enc_logits}
    return dec_logits, {}

  def decoder(self, inputs, z, training, embedded_captions=None):
    """Decodes grayscale representation and masked colors into logits."""
    # (H, W, 512) preprocessing.
    # quantize to 3 bits.
    labels = base_utils.convert_bits(inputs, n_bits_in=8, n_bits_out=3)

    # bin each channel triplet -> (H, W, 3) with 8 possible symbols
    # (H, W, 512)
    labels = base_utils.labels_to_bins(labels, self.num_symbols_per_channel)

    # (H, W) with 512 symbols to (H, W, 512)
    labels = tf.one_hot(labels, depth=self.num_symbols)

    h_dec = self.pixel_embed_layer(labels)
    h_upper = self.outer_decoder((h_dec, z), training=training)
    h_inner = self.inner_decoder((h_dec, h_upper, z), training=training)

    final_activations = h_inner

    # if embedded_captions is not None:
    #   batch_size, height, width, _ = tf.unstack(tf.shape(h_upper))
    #
    #   caption_embedding = tf.cast(embedded_captions["caption"], dtype=tf.float32)
    #   caption_seq = tf.expand_dims(caption_embedding, 1)
    #
    #   # Reshape the final features for cross-attention
    #   h_inner_seq = tf.reshape(h_inner, [batch_size, -1, self.hidden_size])
    #
    #   # --- Apply ONE Conditioning Step: Cross-Attention on h_inner ---
    #   attention_output = self.cross_attention_layer_inner_decode(
    #     query=h_inner_seq,
    #     value=caption_seq,
    #     key=caption_seq,
    #     training=training
    #   )
    #   attention_output = self.cross_attention_dropout_inner_decode(attention_output, training=training)
    #   attention_result = h_inner_seq + attention_output
    #   conditioned_seq = self.cross_attention_norm_inner_decode(attention_result)
    #
    #   final_activations = tf.reshape(conditioned_seq, [batch_size, height, width, self.hidden_size])

    # visualizer.export_logits_channels(logits_tensor=h_inner, output_dir="./logs/h_inner/", prefix="h_inner")
    # Final activation
    activations = self.final_norm(final_activations)
    logits = self.final_dense(activations)

    return tf.expand_dims(logits, axis=-2)

  def image_loss(self, logits, labels):
    """Cross-entropy between the logits and labels."""
    # print(f"labels:{labels.shape}, logits:{logits.shape}")
    height, width = labels.shape[1:3]
    logits = tf.squeeze(logits, axis=-2)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
    loss = tf.reduce_mean(loss, axis=0)
    loss = base_utils.nats_to_bits(tf.reduce_sum(loss))
    return loss / (height * width)

  def compute_loss(self, targets, logits, train_config, training, aux_output=None):
    """Converts targets to coarse colors and computes log-likelihood."""
    downsample = train_config.get('downsample', False)
    downsample_res = train_config.get('downsample_res', 64)
    if downsample:
      labels = targets['targets_%d' % downsample_res]
    else:
      labels = targets['targets']

    if aux_output is None:
      aux_output = {}

    # quantize labels.
    labels = base_utils.convert_bits(labels, n_bits_in=8, n_bits_out=3)

    # bin each channel triplet.
    labels = base_utils.labels_to_bins(labels, self.num_symbols_per_channel)

    loss = self.image_loss(logits, labels)
    enc_logits = aux_output.get('encoder_logits')
    if enc_logits is None:
      return loss, {}

    enc_loss = self.image_loss(enc_logits, labels)
    return loss, {'encoder': enc_loss}

  def get_logits(self, inputs_dict, train_config, training, visualize=False):
    is_downsample = train_config.get('downsample', False)
    downsample_res = train_config.get('downsample_res', 64)
    if is_downsample:
      inputs = inputs_dict['targets_%d' % downsample_res]
    else:
      inputs = inputs_dict['targets']
    embedded_captions = inputs_dict["embedded_captions"]
    result=self(inputs=inputs, embedded_captions=embedded_captions ,training=training)
    # print(f"inputs_dict['label'].numpy()[0].decode('utf-8'):{inputs_dict['label'].numpy()[0].decode('utf-8')}")

    # save input images for visualization
    if visualize:
      # Text 2 color hex
      caption = inputs_dict['captions'].numpy()[0].decode('utf-8')
      print(f"label:{inputs_dict['label'].numpy()[0].decode('utf-8')}")
      print(f"inputs_dict['captions']:{caption}")
      hex_palette=text_to_color_hex.filter_color_words(caption=caption)
      print(f"Generated hex_palette:{hex_palette}")
      if len(hex_palette) != 0:
        coarse_palette_rgb = text_to_color_hex.get_coarse_color_palette_tf()
        target_indices = text_to_color_hex.find_closest_coarse_color_indices(hex_palette, coarse_palette_rgb)
        # print(f"Hex Palette: {hex_palette}")
        print(f"Closest Coarse Color Indices (0-511): {target_indices.numpy()}")

      # Visualize and save the generated image
      output_dir = "./coltran/results/logits/"
      os.makedirs(output_dir, exist_ok=True)
      base_name = inputs_dict['label'].numpy()[0].decode('utf-8')
      filename = f"{base_name}.png"
      output_path = os.path.join(output_dir, filename)
      count = 1
      while os.path.exists(output_path):
        filename = f"{base_name}({count}).png"
        output_path = os.path.join(output_dir, filename)
        count += 1
      logits = result[0]
      probs = tf.squeeze(logits,axis=-2)[0]
      probs = tf.reshape(probs, [-1, probs.shape[-1]])
      # print(f"probs shape after:{probs.shape}")
      # predicted_indices = self.top_p_sample(probs)
      probs=tf.random.categorical(probs, num_samples=1)
      probs=tf.reshape(probs, [self.config.resolution,self.config.resolution])
      # print(f"probs:{probs.shape}")


      plt.imsave(output_path, tf.reshape(self.post_process_image(probs), [self.config.resolution,self.config.resolution, 3]).numpy())
    return result

  def sample(self, gray_cond, captions=None ,mode='argmax',training=False):
    output = {}

    z_gray = self.encoder(gray_cond, training=training)
    conditioned_z=z_gray

    # if captions is not None:
    #   # z_gray = self.blend(text_embedding=captions["caption"], image_features=z_gray,
    #   #                     scale_vector=self.film_scale_generator,shift_vector=self.film_shift_generator)
    #   captions_float = tf.cast(captions["caption"], dtype=tf.float32)
    #   batch_size, height, width, _ = tf.unstack(tf.shape(conditioned_z))
    #
    #   # 1. --- Reshape Inputs for Attention ---
    #   # Reshape image features from (B, H, W, C) to a sequence (B, H*W, C)
    #   image_seq = tf.reshape(z_gray, [batch_size, -1, self.hidden_size])
    #
    #   # Promote the single text embedding to a sequence of length 1
    #   text_seq = tf.expand_dims(captions_float, 1)
    #
    #   # 2. --- Apply Cross-Attention (The Sub-layer) ---
    #   attention_output = self.cross_attention_layer_grayscale_encode(
    #     query=image_seq,
    #     value=text_seq,
    #     key=text_seq,
    #     training=training  # Pass the training flag
    #   )
    #
    #   # 3. --- Apply Dropout ---
    #   attention_output = self.cross_attention_dropout_grayscale_encode(attention_output, training=training)
    #
    #   # 4. --- Apply Residual Connection (The "Add") ---
    #   # This is the most important fix. Add the output back to the input.
    #   attention_result_seq = image_seq + attention_output
    #
    #   # 5. --- Apply Layer Normalization (The "Norm") ---
    #   h_inner_seq = self.cross_attention_norm_grayscale_encode(attention_result_seq)
    #
    #   # 6. --- Reshape back to Image Format ---
    #   conditioned_z = tf.reshape(h_inner_seq, [batch_size, 64, 64, self.hidden_size])

    if self.is_parallel_loss:
      z_logits = self.parallel_dense(conditioned_z)
      parallel_image = tf.argmax(z_logits, axis=-1, output_type=tf.int32)
      parallel_image = self.post_process_image(parallel_image)

      output['parallel'] = parallel_image

    image, proba = self.autoregressive_sample(z_gray=conditioned_z, mode=mode, captions=captions)
    output['auto_%s' % mode] = image
    output['proba'] = proba
    return output

  def autoregressive_sample(self, z_gray, captions=None ,mode='sample'):
    """Generates pixel-by-pixel.

    1. The encoder is run once per-channel.
    2. The outer decoder is run once per-row.
    3. the inner decoder is run once per-pixel.

    The context from the encoder and outer decoder conditions the
    inner decoder. The inner decoder then generates a row, one pixel at a time.

    After generating all pixels in a row, the outer decoder is run to recompute
    context. This condtions the inner decoder, which then generates the next
    row, pixel-by-pixel.

    Args:
      z_gray: grayscale image.
      mode: sample or argmax.

    Returns:
      image: coarse image of shape (B, H, W)
      image_proba: probalities, shape (B, H, W, 512)
    """
    num_filters = self.config.hidden_size
    batch_size, height, width = z_gray.shape[:3]

    # channel_cache[i, j] stores the pixel embedding for row i and col j.
    canvas_shape = (batch_size, height, width, num_filters)
    channel_cache = coltran_layers.Cache(canvas_shape=(height, width))
    init_channel = tf.zeros(shape=canvas_shape)
    init_ind = tf.stack([0, 0])
    channel_cache(inputs=(init_channel, init_ind))

    # upper_context[row_ind] stores context from all previously generated rows.
    upper_context = tf.zeros(shape=canvas_shape)

    # row_cache[0, j] stores the pixel embedding for the column j of the row
    # under generation. After every row is generated, this is rewritten.
    row_cache = coltran_layers.Cache(canvas_shape=(1, width))
    init_row = tf.zeros(shape=(batch_size, 1, width, num_filters))
    row_cache(inputs=(init_row, init_ind))

    pixel_samples, pixel_probas = [], []
    if captions is not None:
      caption_embedding=tf.cast(captions["caption"], dtype=tf.float32)

    for row in range(height):
      row_cond_channel = tf.expand_dims(z_gray[:, row], axis=1)
      row_cond_upper = tf.expand_dims(upper_context[:, row], axis=1)
      row_cache.reset()

      gen_row, proba_row = [], []
      for col in range(width):

        inner_input = (row_cache.cache, row_cond_upper, row_cond_channel)
        # computes output activations at col.
        activations = self.inner_decoder(inner_input, row_ind=row,
                                         training=False)

        pixel_sample, pixel_embed, pixel_proba = self.act_logit_sample_embed(
            activations, col, mode=mode)
        proba_row.append(pixel_proba)
        gen_row.append(pixel_sample)

        # row_cache[:, col] = pixel_embed
        row_cache(inputs=(pixel_embed, tf.stack([0, col])))

        # channel_cache[row, col] = pixel_embed
        channel_cache(inputs=(pixel_embed, tf.stack([row, col])))

      gen_row = tf.stack(gen_row, axis=-1)
      pixel_samples.append(gen_row)
      pixel_probas.append(tf.stack(proba_row, axis=1))
      print(f"pixel_samples len:{len(pixel_samples)}")

      # after a row is generated, recomputes the context for the next row.
      # upper_context[row] = self_attention(channel_cache[:row_index])
      upper_context = self.outer_decoder(
          inputs=(channel_cache.cache, z_gray), training=False)

    image = tf.stack(pixel_samples, axis=1)
    # print(f"Image shape:{image.shape}")
    image = self.post_process_image(image)

    image_proba = tf.stack(pixel_probas, axis=1)
    return image, image_proba

  def act_logit_sample_embed(self, activations, col_ind, mode='sample',adj_embedding=None):
    """Converts activations[col_ind] to the output pixel.

    Activation -> Logit -> Sample -> Embedding.

    Args:
      activations: 5-D Tensor, shape=(batch_size, 1, width, hidden_size)
      col_ind: integer.
      mode: 'sample' or 'argmax'
    Returns:
      pixel_sample: 1-D Tensor, shape=(batch_size, 1, 1)
      pixel_embed: 4-D Tensor, shape=(batch_size, 1, 1, hidden_size)
      pixel_proba: 4-D Tensor, shape=(batch_size, 1, 512)
    """
    batch_size = activations.shape[0]
    pixel_activation = tf.expand_dims(activations[:, :, col_ind], axis=-2)
    final_activations = pixel_activation
    # if adj_embedding is not None:
    #     # Apply the simpler, more stable FiLM conditioning here for adjectives
    #     final_activations = self.blend_adj_inner(adj_embedding, pixel_activation)

    pixel_logits = self.final_dense(self.final_norm(final_activations))
    pixel_logits = tf.squeeze(pixel_logits, axis=[1, 2])
    pixel_proba = tf.nn.softmax(pixel_logits, axis=-1)

    if mode == 'sample':
      pixel_sample = tf.random.categorical(
          pixel_logits, num_samples=1, dtype=tf.int32)
      pixel_sample = tf.squeeze(pixel_sample, axis=-1)
    elif mode == 'argmax':
      pixel_sample = tf.argmax(pixel_logits, axis=-1, output_type=tf.int32)
    pixel_sample_expand = tf.reshape(pixel_sample, [batch_size, 1, 1])
    pixel_one_hot = tf.one_hot(pixel_sample_expand, depth=self.num_symbols)
    pixel_embed = self.pixel_embed_layer(pixel_one_hot)
    return pixel_sample, pixel_embed, pixel_proba

  def post_process_image(self, image):
    """Post process image of size (H, W, 512) to a coarse RGB image."""
    image = base_utils.bins_to_labels(
        image, num_symbols_per_channel=self.num_symbols_per_channel)
    image = base_utils.convert_bits(image, n_bits_in=3, n_bits_out=8)
    image = tf.cast(image, dtype=tf.uint8)
    return image

  def get_color_palette(self,n_bits=3):
    """
    Generates a color palette for 3-bit RGB quantization (8 values per channel, 512 total colors).
    Blend with noun embedding
    Returns:
        palette: tf.Tensor of shape [512, 3], dtype=tf.int32.
                  Each row is an RGB color in [0, 255].
    """
    num_values = 2 ** n_bits  # 8 values per channel (3 bits)
    # Linearly spaced values from 0 to 255 (8-bit) for each channel
    channel_values = tf.linspace(0.0, 255.0, num_values)
    channel_values = tf.round(channel_values)  # Round to nearest integer
    channel_values = tf.cast(channel_values, tf.int32)  # [0, 36, 73, ..., 255]

    # Generate all combinations of R, G, B values
    R, G, B = tf.meshgrid(channel_values, channel_values, channel_values, indexing='ij')
    palette = tf.stack([R, G, B], axis=-1)  # Shape: [8, 8, 8, 3]

    # Flatten to [512, 3] and return
    return tf.reshape(palette, (-1, 3))

  # def blend_adj_inner(self, adj_embedding, image_features):
  #   scale_vec = tf.nn.sigmoid(self.color_scale_generator_inner(adj_embedding)) + 0.5
  #   shift_vec = self.color_shift_generator_inner(adj_embedding)
  #
  #   batch_size = tf.shape(image_features)[0]
  #   scale = tf.reshape(scale_vec, (batch_size, 1, 1, self.hidden_size))
  #   shift = tf.reshape(shift_vec, (batch_size, 1, 1, self.hidden_size))
  #
  #   return image_features * scale + shift

  def top_p_sample(self,logits, p=0.95, temperature=1.0):
    """
    Performs Top-P (Nucleus) sampling.
    """
    scaled_logits = logits / max(temperature, 1e-9)

    # Convert logits to probabilities
    probabilities = tf.nn.softmax(scaled_logits, axis=-1)

    # Sort probabilities in descending order
    sorted_indices = tf.argsort(probabilities, axis=-1, direction='DESCENDING')
    sorted_probs = tf.gather(probabilities, sorted_indices, batch_dims=1)

    # Calculate cumulative probabilities
    cumulative_probs = tf.math.cumsum(sorted_probs, axis=-1)

    # Find the indices to remove (those that are beyond the cumulative p)
    indices_to_remove = cumulative_probs > p
    # Shift to the right to keep the first one that exceeds p
    indices_to_remove = tf.concat([tf.zeros_like(indices_to_remove[:, :1]), indices_to_remove[:, :-1]], axis=-1)

    # Create a mask to set the probabilities of removed tokens to 0
    # We need to unsort this mask to apply it to the original probabilities
    inverse_indices = tf.argsort(sorted_indices, axis=-1)
    mask = tf.gather(indices_to_remove, inverse_indices, batch_dims=1)

    probabilities = tf.where(mask, 0.0, probabilities)

    # Re-normalize
    probabilities /= tf.reduce_sum(probabilities, axis=-1, keepdims=True)

    # Sample from the modified distribution
    sample = tf.random.categorical(tf.math.log(probabilities + 1e-9), num_samples=1, dtype=tf.int32)
    return tf.squeeze(sample, axis=-1)