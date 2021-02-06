from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend

#ARCHITECTURE COLOR RECONTSRUCTION
def createUpdatedModel():
    # HyperParameters Initialization :
    kernelSize = 3
    latentDim = 256  # Simply number of units that is the Input of the Dense Layer
    stride = 2
    # layerFilters = [64, 128, 256]
    inputShape = (32, 32, 1)  # grayscale dataset shape

    # Input Layer:
    inputs = Input(shape=inputShape, name='EncoderInputLayer')
    model = inputs

    # Encoder Construction 3 conv layers with 64,128,256:
    model = Conv2D(64, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)
    model = Conv2D(128, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)
    model = Conv2D(256, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)

    # Apparently MaxPooling decreases the quality of the results(val_acc) so we don't use it in the encoder
    # currentModelShape
    shape = backend.int_shape(model)

    # Flattening the model:
    model = Flatten()(model)
    #
    latent = Dense(latentDim)(model)

    encoder = Model(inputs, latent, name='EncoderModel')

    latentInputs = Input(shape=(latentDim,), name='DecoderInputLayer')
    model = Dense(shape[1] * shape[2] * shape[3])(latentInputs)
    model = Reshape((shape[1], shape[2], shape[3]))(model)

    # We create 3 conv2dTranspose layers in respect with the conv2d 3:3 symmetry
    model = Conv2DTranspose(256, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)
    model = Conv2DTranspose(128, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)
    model = Conv2DTranspose(64, kernel_size=kernelSize, strides=stride, activation='relu', padding='same')(model)

    # Output Layer
    output = Conv2DTranspose(3, kernel_size=kernelSize, activation='sigmoid', padding='same',
                             name='DecoderOutputLayer')(model)

    decoder = Model(latentInputs, output, name='DecoderModel')

    # instantiate autoencoder model
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='AutoEncoderModel')

    return autoencoder
