import numpy as np
import nibabel as nib
import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, BatchNormalization, Activation, \
    UpSampling2D, MaxPooling2D, Conv2DTranspose, Dropout, Concatenate, Input
from keras.models import Model
import random
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import pandas as pd

def image_proc(filepath):
    """ Data loader (*.nii)
    :param filepath: file path
    :return: 2D array images
    """
    img_data0 = np.zeros((96, 96, 1), dtype='float32')# np.zeros((96, 96, 1), dtype='float32')
    img_data = []
    for item in tqdm(sorted(filepath), desc='Processing'):
        # loading images
        img = nib.load(item).get_fdata()
        # # Crop to get the brain region (along z-axis and x & y axes)
        ind = np.where(img > 0)
        ind_min, ind_max = min(ind[2]), max(ind[2])
        ind_mid = round((ind_min + ind_max) / 2)
        img = img[8:232,8:232,ind_mid-32:ind_mid+32]   # to have 224 x 224 x 64 dim.
        # resize
        print('SHAPE', img.ndim)  # Add this line before zoom to check the dimensionality
        img = img[..., 0] 
        print('SHAPE', img.ndim)
        img = zoom(img, (0.428, 0.428, 1))   # to have 96 x 96 x 64 dim.
        # Normalize using zero mean and unit variance method & scale to 0-1 range
        img = ((img - img.mean()) / img.std())
        img = ((img - img.min()) / (img.max() - img.min()))  # Scale to 0-1 range
        # Convert 3D images to 2D image slices
        img_data0 = np.concatenate((img_data0, img), axis=2)
    img_data0 = np.moveaxis(img_data0, [2], [0])
    return np.array(img_data0[1::,:,:]).astype('float32')

# Usage example:
# image_proc(['path/to/image1.nii', 'path/to/image2.nii'])


def conv_block(input, num_filters):
    """ Convolutional Layers """
    x = Conv2D(num_filters, 3, kernel_initializer='he_uniform', padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, kernel_initializer='he_uniform', padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(input, num_filters):
    """ Encoder Block """
    x = conv_block(input, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(input)
    # x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_2DUNet_model_v1(input_shape):
    """ U-NET Architecture """
    inputs = Input(input_shape, dtype='float32')
    ini_numb_of_filters = 16

    """ Eecoder 1, 2, 3, 4 """
    s1, p1 = encoder_block(inputs, ini_numb_of_filters)
    s2, p2 = encoder_block(p1, ini_numb_of_filters * 2)
    s3, p3 = encoder_block(p2, ini_numb_of_filters * 4)
    s4, p4 = encoder_block(p3, ini_numb_of_filters * 8)

    """ Bridge """
    b1 = conv_block(p4, ini_numb_of_filters * 16)

    """ Decoder 1, 2, 3, 4 """
    d1 = decoder_block(b1, s4, ini_numb_of_filters * 8)
    d2 = decoder_block(d1, s3, ini_numb_of_filters * 4)
    d3 = decoder_block(d2, s2, ini_numb_of_filters * 2)
    d4 = decoder_block(d3, s1, ini_numb_of_filters)

    """ Outputs  """
    outputs = Conv2D(1, 1, padding="same", activation="linear")(d4)

    from keras.optimizers import Adam
    learning_rate = 0.001
    optimizer = Adam(learning_rate)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy', 'mae'])
    return model


def plot_learning_curve(filepath):
    df = pd.read_csv(filepath)
    df_x, df_yt, df_yv = df.values[:, 0], df.values[:, 2], df.values[:, 5]
    plt.figure(figsize=(5, 4))
    plt.plot(df_x, df_yt)
    plt.plot(df_x, df_yv)
    # plt.title('average training loss and validation loss')
    plt.ylabel('mean-squared error', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['training loss', 'validation loss'], fontsize=14, loc='upper right')
    plt.show()
    return

# plot_learning_curve("MR_Synth_2D_logs.csv")
