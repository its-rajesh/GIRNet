import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    LeakyReLU, Conv2D, Conv2DTranspose, MaxPooling2D,
    BatchNormalization, Dropout, Input, concatenate, Reshape
)
from tensorflow.keras.models import Model
from spektral.layers import GATConv


class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.channels = channels
        self.gat = GATConv(
            channels=channels,
            attn_heads=1,
            concat_heads=True,
            dropout_rate=0.2,
            activation=None,
        )
        self.leaky_relu = LeakyReLU(0.3)

    def call(self, inputs, training=True):
        x, adj = inputs
        out = self.gat([x, adj], training=training)
        return self.leaky_relu(out)


def build_encoder(input_tensor):
    def down_block(x, filters, dilation):
        x = Conv2D(filters, (4, 4), padding="same", dilation_rate=dilation)(x)
        x = Conv2D(filters, (4, 4), padding="same", dilation_rate=dilation)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        x = MaxPooling2D((1, 2))(x)
        return Dropout(0.1)(x)

    ds1 = down_block(input_tensor, 16, 1)
    ds2 = down_block(ds1, 32, 2)
    ds3 = down_block(ds2, 64, 4)
    ds4 = down_block(ds3, 128, 16)

    return ds4, ds3, ds2, ds1


def build_bottleneck(x):
    x = Conv2D(256, (4, 4), padding="same")(x)
    x = Conv2D(256, (4, 4), padding="same")(x)
    x = BatchNormalization()(x)
    return LeakyReLU(0.3)(x)


def build_decoder(x, ds4, ds3, ds2, ds1):
    def up_block(x, skip, filters):
        x = Conv2DTranspose(filters, (4, 4), strides=(1, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        x = concatenate([x, skip])
        x = Conv2D(filters, (4, 4), padding="same")(x)
        x = Conv2D(filters, (4, 4), padding="same")(x)
        return LeakyReLU(0.3)(x)

    x = Conv2DTranspose(128, (4, 4), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.3)(x)
    x = concatenate([x, ds4])
    x = Conv2D(128, (4, 4), padding="same")(x)
    x = Conv2D(128, (4, 4), padding="same")(x)
    x = LeakyReLU(0.3)(x)

    x = up_block(x, ds3, 64)
    x = up_block(x, ds2, 32)
    x = up_block(x, ds1, 16)
    return x


def cosine_graph_adj(features, threshold=0.6):
    sim = tf.matmul(features, features, transpose_b=True)
    adj = tf.cast(sim >= threshold, tf.float32)
    adj = adj - tf.linalg.diag(tf.linalg.diag_part(adj))
    return adj


def build_girnet_model(input_shape, channels=4, threshold=0.6):
    inp = Input(shape=(channels, input_shape, 1))
    ds4, ds3, ds2, ds1 = build_encoder(inp)

    reshaped = Reshape((-1, ds4.shape[-1]))(ds4)
    adj = tf.keras.layers.Lambda(lambda x: cosine_graph_adj(x, threshold))(reshaped)

    gat_out = GraphAttentionLayer(channels=ds4.shape[-1])([reshaped, adj])
    reverted = Reshape((ds4.shape[1], ds4.shape[2], ds4.shape[3]))(gat_out)

    merged = tf.keras.layers.Multiply()([reverted, ds4])
    bottleneck_out = build_bottleneck(merged)
    decoded = build_decoder(bottleneck_out, ds4, ds3, ds2, ds1)

    x = Conv2DTranspose(1, (4, 4), strides=(1, 2), padding="same")(decoded)
    out = LeakyReLU(0.3)(x)

    return Model(inputs=inp, outputs=out, name="GIRNet")


def train_model(Xtrain, Ytrain, dim, batch_size=1, epochs=30, threshold=0.6, save_path='GAT_weights.h5', save_full=False):
    model = build_girnet_model(dim, threshold=threshold)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, shuffle=True)

    if save_full:
        model.save(save_path)
    else:
        model.save_weights(save_path)
    print(f"Model {'saved' if save_full else 'weights saved'} to {save_path}")


################################################################
#                        DRIVER CODE                           #
################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GIRNet for interference reduction")
    parser.add_argument("--data_clean", type=str, required=True, help="Path to clean Ytrain.npy")
    parser.add_argument("--data_mixed", type=str, required=True, help="Path to CM_Xtrain.npy")
    parser.add_argument("--dim", type=int, default=1024, help="Input frame length")
    parser.add_argument("--n_examples", type=int, default=2, help="Number of training examples to load")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="GAT_weights.h5", help="Path to save the model/weights")
    parser.add_argument("--save_full_model", action="store_true", help="If set, saves full model instead of weights")
    parser.add_argument("--threshold", type=float, default=0.6, help="Cosine similarity threshold for adjacency")

    args = parser.parse_args()

    print("Loading data...")
    Xtrain = np.load(args.data_mixed)[:args.n_examples, :, :args.dim]
    Ytrain = np.load(args.data_clean)[:args.n_examples, :, :args.dim]

    print(f"Training data shape: X={Xtrain.shape}, Y={Ytrain.shape}")
    train_model(
        Xtrain,
        Ytrain,
        dim=args.dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        threshold=args.threshold,
        save_path=args.save_path,
        save_full=args.save_full_model
    )
