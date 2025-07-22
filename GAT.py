from spektral.layers import GATConv
from tensorflow.keras.layers import LeakyReLU, Conv2D, concatenate, Conv2DTranspose, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Input, Dense, Reshape
from tensorflow.keras.models import Model, load_model, save_model
import tensorflow as tf
from tensorflow.keras.layers import Layer
from spektral.utils import normalized_adjacency
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from spektral.layers import GATConv



class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.channels = channels
        self.gat_layer = GATConv(
            channels=channels,
            attn_heads=1,
            concat_heads=True,
            dropout_rate=0.2,
            activation=None,
        )
        self.leaky_relu = LeakyReLU(alpha=0.3)

    def call(self, inputs, training=True):
        x, a = inputs  # Features and adjacency matrix
        output = self.gat_layer([x, a], training=training)
        output = self.leaky_relu(output)
        return output

    def get_config(self):
        config = {'channels': self.channels}
        base_config = super(GraphAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def encoder(input_shape):

    #Downsampling block 1
    x = Conv2D(16, (4, 4), padding = "same", dilation_rate=1)(input_shape)
    x = Conv2D(16, (4, 4), padding = "same", dilation_rate=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds1 = Dropout(0.1)(x)

    #Downsampling block 2
    x = Conv2D(32, (4, 4), padding = "same", dilation_rate=2)(ds1)
    x = Conv2D(32, (4, 4), padding = "same", dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds2 = Dropout(0.1)(x)

    #Downsampling block 3
    x = Conv2D(64, (4, 4), padding = "same", dilation_rate=4)(ds2)
    x = Conv2D(64, (4, 4), padding = "same", dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds3 = Dropout(0.1)(x)

    #Downsampling block 4
    x = Conv2D(128, (4, 4), padding = "same", dilation_rate=16)(ds3)
    x = Conv2D(128, (4, 4), padding = "same", dilation_rate=16)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds4 = Dropout(0.1)(x)

    return ds4, ds3, ds2, ds1


def bottleneck(encoder_output):
    # Bottleneck layer
    x = Conv2D(256, (4, 4), padding = "same")(encoder_output)
    x = Conv2D(256, (4, 4), padding = "same")(x)
    x = BatchNormalization()(x)
    out = LeakyReLU(alpha=0.3)(x)
    
    return out


def decoder(bottleneck_output, ds4, ds3, ds2, ds1):

    #Upsampling Block 4
    x = Conv2DTranspose(128, (4, 4), (1, 1), padding = "same")(bottleneck_output)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds4])

    x = Conv2D(128, (4, 4), padding = "same")(x)
    x = Conv2D(128, (4, 4), padding = "same")(x)
    up4 = LeakyReLU(alpha=0.3)(x)

    #Upsampling Block 3
    x = Conv2DTranspose(64, (4, 4), (1, 2), padding = "same")(up4)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds3])

    x = Conv2D(64, (4, 4), padding = "same")(x)
    x = Conv2D(64, (4, 4), padding = "same")(x)
    up3 = LeakyReLU(alpha=0.3)(x)

    #Upsampling Block 2
    x = Conv2DTranspose(32, (4, 4), (1, 2), padding = "same")(up3)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds2])

    x = Conv2D(32, (4, 4), padding = "same")(x)
    x = Conv2D(32, (4, 4), padding = "same")(x)
    up2 = LeakyReLU(alpha=0.3)(x)

    #Upsampling Block 1
    x = Conv2DTranspose(16, (4, 4), (1, 2), padding = "same")(up2)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds1])

    x = Conv2D(32, (4, 4), padding = "same")(x)
    x = Conv2D(32, (4, 4), padding = "same")(x)
    up1 = LeakyReLU(alpha=0.3)(x)

    return up1


def iwaveUnet(Xtrain, Ytrain, batch_size, epochs, dim):

    channels = 4
    samples = dim #220448
    input_shape = Input(shape=(channels, samples, 1))

    # ENCODER ------------------------------------------------------------
    ds4, ds3, ds2, ds1 = encoder(input_shape)
    #reshaped = tf.reshape(ds4, shape=(-1, 4, ds4.shape[2]*ds4.shape[3]))

    # ATTENTION GRAPH NEURAL NETWORK -------------------------------------
    num_nodes = ds4.shape[1]*ds4.shape[2]
    num_features = ds4.shape[3]

    reshaped = tf.reshape(ds4, shape=(-1, ds4.shape[1]*ds4.shape[2], ds4.shape[3]))
    #A = np.ones((num_nodes, num_nodes))

    # Calculate cosine similarity
    cosine_sim_matrix = tf.matmul(reshaped, reshaped, transpose_b=True)
    # Define a threshold for cosine similarity
    threshold = 0.6
    # Create the adjacency matrix using a binary threshold
    A = tf.cast(cosine_sim_matrix >= threshold, tf.float32)
    # Fill the diagonal with zeros (optional)
    A = A - tf.linalg.diag(tf.linalg.diag_part(A))

    GATLayer = GraphAttentionLayer(channels=num_features)

    #A = tf.constant(A, dtype=tf.float32)
    #print(type(A), type(reshaped), A)
    graph_out = GATLayer([reshaped, A],  training=True)
    #print(A.shape, reshaped.shape)
   
    # BACK TO UNET -------------------------------------------------------
    reverted_output = Reshape(target_shape=(4, ds4.shape[2], ds4.shape[3]))(graph_out)
    hadamard_product = tf.keras.layers.multiply([reverted_output, ds4])

    # BOTTLE NECK --------------------------------------------------------
    bottleneck_output = bottleneck(hadamard_product)

    # DECODER ------------------------------------------------------------
    decoder_output = decoder(bottleneck_output, ds4, ds3, ds2, ds1)

    x = Conv2DTranspose(1, (4, 4), (1, 2), padding="same")(decoder_output)
    output = LeakyReLU(alpha=0.3)(x)

    iwaveunet = Model(input_shape, output, name="GAT")

    iwaveunet.compile(optimizer='adam', loss='mse')

    iwaveunet.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, shuffle=True)
    print(iwaveunet.summary())
    #iwaveunet.save('GAT.h5')
    iwaveunet.save_weights('GAT_weights.h5')


    print('Model Saved')



################################################################
#                        DRIVER CODE                           #
################################################################

if __name__ == "__main__":

    dpath = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/LM/"
    dpath2 = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/CM/"

    print('Loading Files...')

    Xtrain1 = np.load(dpath2+'CM_Xtrain.npy')
    #Xtrain2 = np.load(dpath2+'CM_Xtrain2.npy')
    #Xtrain3 = np.load(dpath2+'CM_Xtrain3.npy')
    Ytrain = np.load(dpath+'Ytrain.npy')

    dim = 1024 #220448 #Note
    #Xtrain1 = Xtrain1[:, :, :dim]

    #Xtrain = np.vstack([Xtrain1, Xtrain2, Xtrain3])
    #Ytrain = np.vstack([Ytrain, Ytrain, Ytrain])

    Xtrain = Xtrain1[:2, :, :dim] #NOTE HERE
    Ytrain = Ytrain[:2, :, :dim]

    print(Xtrain.shape, Ytrain.shape)

    print('Training....')
    batch_size = 1
    epochs = 1
    iwaveUnet(Xtrain, Ytrain, batch_size, epochs, dim)
