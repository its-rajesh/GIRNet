from spektral.layers import GATConv
from tensorflow.keras.layers import LeakyReLU, Conv2D, concatenate, Conv2DTranspose, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Input, Dense, Reshape
from tensorflow.keras.models import Model, load_model, save_model
import tensorflow as tf
from tensorflow.keras.layers import Layer
from spektral.utils import normalized_adjacency
from scipy.sparse import csr_matrix
import numpy as np


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



channels = 4
samples = 220448
input_shape = Input(shape=(channels, samples, 1))

# ENCODER ------------------------------------------------------------
ds4, ds3, ds2, ds1 = encoder(input_shape)
#reshaped = tf.reshape(ds4, shape=(-1, 4, ds4.shape[2]*ds4.shape[3]))

# ATTENTION GRAPH NEURAL NETWORK -------------------------------------
num_nodes = ds4.shape[1]*ds4.shape[2]
num_features = ds4.shape[3]

reshaped = tf.reshape(ds4, shape=(-1, ds4.shape[1]*ds4.shape[2], ds4.shape[3]))
A = np.ones((num_nodes, num_nodes))

GATLayer = GraphAttentionLayer(channels=num_features)

A = tf.constant(A, dtype=tf.float32)
print(type(A), type(reshaped))
graph_out = GATLayer([reshaped, A],  training=True)
print(A.shape, reshaped.shape)

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

print('Model Compiled')

iwaveunet.load_weights('/scratch/rajeshr.scee.iitmandi/dfUNet/GAT_weights_20to25.h5')

print(iwaveunet.summary())


from tqdm import tqdm
import soundfile as sf
import pandas as pd
import os
import numpy as np

path = "/scratch/rajeshr.scee.iitmandi/dfUNet/"
dpath = "/scratch/rajeshr.scee.iitmandi/Dataset/MUSDBHQ/LMTest/"

print('(1/4) Loading Files...')
# LINEAR MIXTURES
y1 = np.load(dpath+'ytest.npy')
y2 = np.load(dpath+'ytest_lh.npy')
Ytest = np.vstack([y1, y2])

# CONVOLUTE MIXTURES
dpath2 = "/scratch/rajeshr.scee.iitmandi/Dataset/MUSDBHQ/CMTest/"
Xtest = np.load(dpath2+'CM_Xtest.npy')

Xtest = Xtest[:, :, :220448]
Ytest = Ytest[:, :, :220448]
print(Xtest.shape, Ytest.shape)

#i = 22 
#old = [64, 76, 99, 87, 77, 42]

print("Evaluating..")
for i in [10, 41, 64, 75, 87, 31]:
    model = tf.keras.models.load_model('GAT_weights_20to25.h5')

    predictions = model.predict(np.expand_dims(Xtest[i], axis=0))
    predictions = np.squeeze(predictions)
    print(predictions.shape)

    os.makedirs(path+'/Outputs/{}/'.format(i),exist_ok=True)

    sf.write(path+'/Outputs/{}/bvocal.wav'.format(i), Xtest[i][0], 22050)
    sf.write(path+'/Outputs/{}/bbass.wav'.format(i), Xtest[i][1], 22050)
    sf.write(path+'/Outputs/{}/bdrums.wav'.format(i), Xtest[i][2], 22050)
    sf.write(path+'/Outputs/{}/bother.wav'.format(i), Xtest[i][3], 22050)

    sf.write(path+'/Outputs/{}/tvocal.wav'.format(i), Ytest[i][0], 22050)
    sf.write(path+'/Outputs/{}/tbass.wav'.format(i), Ytest[i][1], 22050)
    sf.write(path+'/Outputs/{}/tdrums.wav'.format(i), Ytest[i][2], 22050)
    sf.write(path+'/Outputs/{}/tother.wav'.format(i), Ytest[i][3], 22050)

    sf.write(path+'/Outputs/{}/pvocal.wav'.format(i), predictions[0], 22050)
    sf.write(path+'/Outputs/{}/pbass.wav'.format(i), predictions[1], 22050)
    sf.write(path+'/Outputs/{}/pdrums.wav'.format(i), predictions[2], 22050)
    sf.write(path+'/Outputs/{}/pother.wav'.format(i), predictions[3], 22050)

print('DONE')
