import tensorflow as tf
# from tensorflow.python.keras.utils.vis_utils import plot_model

# Define the neural network model parameters
IMG_PATCH_SHAPE = 128
FEATURE_DIM = 64
MODEL_DEPTH = 12
RESIDUAL_BLOCK_DEPTH = 4
NUM_RESIDUAL_BLOCKS = 2
CONV_KERNEL_SIZE = [3, 3]
BASE_LR = 1e-4

def residualBlock(h, width, kernelSize, depth):
    h_in = h
    for i in range(depth):
        h = tf.keras.layers.Conv2D(width, kernelSize, padding='same', activation='relu')(h)
    return h_in + h

def buildModelFullMetadata():
    """
    Setup noise source estimator with full set of camera metadata.
    :return: Initialized model.
    """
    
    # Setup the NN model
    x = tf.keras.Input(name="x", shape=(IMG_PATCH_SHAPE, IMG_PATCH_SHAPE, 1), dtype=tf.dtypes.float32)
    h = x
    for idx in range(0, NUM_RESIDUAL_BLOCKS):
        h = tf.keras.layers.Conv2D(FEATURE_DIM, CONV_KERNEL_SIZE, padding='same',  activation='relu')(h)
        h = residualBlock(h, FEATURE_DIM, CONV_KERNEL_SIZE, RESIDUAL_BLOCK_DEPTH)
        
    gain = tf.keras.Input(name="gain", shape=(1), dtype=tf.dtypes.float32)
    temp = tf.keras.Input(name="temp", shape=(1), dtype=tf.dtypes.float32)
    texp = tf.keras.Input(name="texp", shape=(1), dtype=tf.dtypes.float32)
    sensorType = tf.keras.Input(name="sensorType", shape=(1), dtype=tf.dtypes.float32)
    fullWellSize = tf.keras.Input(name="fullWellSize", shape=(1), dtype=tf.dtypes.float32)
    sensorPixelSize = tf.keras.Input(name="sensorPixelSize", shape=(1), dtype=tf.dtypes.float32)
    darkSignalFoM = tf.keras.Input(name="darkSignalFoM", shape=(1), dtype=tf.dtypes.float32)
    pixelClock = tf.keras.Input(name="pixelClock", shape=(1), dtype=tf.dtypes.float32)
    senseNodeGain = tf.keras.Input(name="senseNodeGain", shape=(1), dtype=tf.dtypes.float32)
    thermalWhiteNoise = tf.keras.Input(name="thermalWhiteNoise", shape=(1), dtype=tf.dtypes.float32)
    senseNodeResetFactor = tf.keras.Input(name="senseNodeResetFactor", shape=(1), dtype=tf.dtypes.float32)
        
    # Photon Noise branch
    h1 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h1 = tf.keras.layers.Concatenate()([h1, gain, fullWellSize])
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(1, activation='linear', name="photonNoiseOutput")(h1)
    
    # DCSN branch
    h2 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h2 = tf.keras.layers.Concatenate()([h2, gain, temp, texp, fullWellSize, sensorPixelSize, darkSignalFoM])
    h2 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h2)
    h2 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h2)
    h2 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h2)
    h2 = tf.keras.layers.Dense(1, activation='linear', name="dcsnOutput")(h2)
    
    # Readout branch
    h3 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h3 = tf.keras.layers.Concatenate()([h3, gain, temp, sensorType, fullWellSize, pixelClock, senseNodeGain, thermalWhiteNoise, senseNodeResetFactor])
    h3 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h3)
    h3 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h3)
    h3 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h3)
    h3 = tf.keras.layers.Dense(1, activation='linear', name="readoutNoiseOutput")(h3)
    
    # ResidualNoise branch
    h4 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h4 = tf.keras.layers.Concatenate()([h1, h2, h3, h4])
    h4 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h4)
    h4 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h4)
    h4 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h4)
    h4 = tf.keras.layers.Dense(1, activation='linear', name="residualNoiseOutput")(h4)
    
    M = tf.keras.Model(
            [x, gain, temp, texp, sensorType, fullWellSize, sensorPixelSize, darkSignalFoM, pixelClock, senseNodeGain, thermalWhiteNoise, senseNodeResetFactor], 
            [h1, h2, h3, h4], 
            name="tfv2_keras_model"
        )
    
    # M.summary()
    # plot_model(M, to_file='model_plot5.png', show_shapes=True, show_layer_names=True)
    
    return M

def buildModelMinimalMetadata():
    """
    Setup noise source estimator with minimal set of camera metadata.
    :return: Initialized model.
    """
    
    # Setup the NN model
    x = tf.keras.Input(name="x", shape=(IMG_PATCH_SHAPE, IMG_PATCH_SHAPE, 1), dtype=tf.dtypes.float32)
    h = x
    for idx in range(0, NUM_RESIDUAL_BLOCKS):
        h = tf.keras.layers.Conv2D(FEATURE_DIM, CONV_KERNEL_SIZE, padding='same',  activation='relu')(h)
        h = residualBlock(h, FEATURE_DIM, CONV_KERNEL_SIZE, RESIDUAL_BLOCK_DEPTH)
        
    gain = tf.keras.Input(name="gain", shape=(1), dtype=tf.dtypes.float32)
    temp = tf.keras.Input(name="temp", shape=(1), dtype=tf.dtypes.float32)
    texp = tf.keras.Input(name="texp", shape=(1), dtype=tf.dtypes.float32)
        
    # Photon Noise branch
    h1 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h1 = tf.keras.layers.Concatenate()([h1, gain])
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(1, activation='linear', name="photonNoiseOutput")(h1)
    
    # DCSN branch
    h2 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h2 = tf.keras.layers.Concatenate()([h2, gain, temp, texp])
    h2 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h2)
    h2 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h2)
    h2 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h2)
    h2 = tf.keras.layers.Dense(1, activation='linear', name="dcsnOutput")(h2)
    
    # Readout branch
    h3 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h3 = tf.keras.layers.Concatenate()([h3, gain, temp])
    h3 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h3)
    h3 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h3)
    h3 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h3)
    h3 = tf.keras.layers.Dense(1, activation='linear', name="readoutNoiseOutput")(h3)
    
    # ResidualNoise branch
    h4 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h4 = tf.keras.layers.Concatenate()([h1, h2, h3, h4])
    h4 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h4)
    h4 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h4)
    h4 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h4)
    h4 = tf.keras.layers.Dense(1, activation='linear', name="residualNoiseOutput")(h4)
    
    M = tf.keras.Model(
            [x, gain, temp, texp], 
            [h1, h2, h3, h4], 
            name="tfv2_keras_model"
        )
    # M.summary()
    # plot_model(M, to_file='model_plot5.png', show_shapes=True, show_layer_names=True)
    
    return M

def buildModelWithoutMetadata():
    """
    Setup noise source estimator without access to camera metadata.
    :return: Initialized model.
    """
    
    # Setup the NN model
    x = tf.keras.Input(name="x", shape=(IMG_PATCH_SHAPE, IMG_PATCH_SHAPE, 1), dtype=tf.dtypes.float32)
    h = x
    for idx in range(0, NUM_RESIDUAL_BLOCKS):
        h = tf.keras.layers.Conv2D(FEATURE_DIM, CONV_KERNEL_SIZE, padding='same',  activation='relu')(h)
        h = residualBlock(h, FEATURE_DIM, CONV_KERNEL_SIZE, RESIDUAL_BLOCK_DEPTH)
        
    # Photon Noise branch
    h1 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(1, activation='linear', name="photonNoiseOutput")(h1)
    
    # DCSN branch
    h2 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h2 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h2)
    h2 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h2)
    h2 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h2)
    h2 = tf.keras.layers.Dense(1, activation='linear', name="dcsnOutput")(h2)
    
    # Readout branch
    h3 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h3 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h3)
    h3 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h3)
    h3 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h3)
    h3 = tf.keras.layers.Dense(1, activation='linear', name="readoutNoiseOutput")(h3)
    
    # ResidualNoise branch
    h4 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h4 = tf.keras.layers.Concatenate()([h1, h2, h3, h4])
    h4 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h4)
    h4 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h4)
    h4 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h4)
    h4 = tf.keras.layers.Dense(1, activation='linear', name="residualNoiseOutput")(h4)
    
    M = tf.keras.Model(
            [x], 
            [h1, h2, h3, h4], 
            name="tfv2_keras_model"
        )
    M.summary()
    # plot_model(M, to_file='model_plot5.png', show_shapes=True, show_layer_names=True)
    
    return M

def buildModelBaseline():
    """
    Setup the baselinen noise source estimator.
    :return: Initialized model.
    """
    
    # Setup the NN model
    x = tf.keras.Input(name="x", shape=(IMG_PATCH_SHAPE, IMG_PATCH_SHAPE, 1), dtype=tf.dtypes.float32)
    h = x
    for idx in range(0, NUM_RESIDUAL_BLOCKS):
        h = tf.keras.layers.Conv2D(FEATURE_DIM, CONV_KERNEL_SIZE, padding='same', activation='relu')(h)
        h = residualBlock(h, FEATURE_DIM, CONV_KERNEL_SIZE, RESIDUAL_BLOCK_DEPTH)
        
    # Total noise branch
    h1 = tf.keras.layers.GlobalMaxPooling2D()(h)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 2, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 4, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(FEATURE_DIM / 8, activation='relu')(h1)
    h1 = tf.keras.layers.Dense(1, activation='linear')(h1)
    
    M = tf.keras.Model(
            [x], 
            [h1], 
            name="tfv2_keras_model"
        )
    M.summary()
    # plot_model(M, to_file='model_plot5.png', show_shapes=True, show_layer_names=True)
    
    return M

def loadModel(modelDir, modelType):
    """
    Setup a raw model and load trained weights from the modelDir.
    :param modelDir: Path to weights of trained model.
    :param modelType: String to specify the desired model type to load.
    :return: If weights could be found in modelDir: Model with trained weights. Otherwise: Untrained model.
    """
    
    if modelType == "fullMetadata":
        model = buildModelFullMetadata()
    elif modelType == "minimalMetadata":
        model = buildModelMinimalMetadata()
    elif modelType == "withoutMetadata":
        model = buildModelWithoutMetadata()
    elif modelType == "baseline":
        model = buildModelBaseline()
    else:
        raise Exception("Unknown model type: " + modelType  + ". Please specifiy a model type from the following ones: baseline, withoutMetadata, minimalMetadata or fullMetadata.")
        
    try:
        model.load_weights(modelDir)
        print("Successfully load model weights for model type " + modelType + ".")
    except:
        print("Warning: No model weights found. Continue with untrained model of type " + modelType + ".")  
    
    losses = {
        "photonNoiseOutput": tf.keras.losses.MeanSquaredError(),
        "dcsnOutput": tf.keras.losses.MeanSquaredError(), 
        "readoutNoiseOutput" : tf.keras.losses.MeanSquaredError(),
        "residualNoiseOutput" : tf.keras.losses.MeanSquaredError()
    }
    lossWeights = {
        "photonNoiseOutput": 1.0, 
        "dcsnOutput": 1.0, 
        "readoutNoiseOutput" : 1.0,
        "residualNoiseOutput" : 1.0}

    # Note that this resets the learning rate if you continue training from a checkpoint.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR),
        loss=losses,
        loss_weights=lossWeights,
        metrics=["mse"]
    )
    
    return model