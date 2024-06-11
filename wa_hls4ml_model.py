from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l1


def create_model_classification():
    model = Sequential()
    # d_in	d_out prec rf strategy (one-hot encoded)
    model.add(Dense(128, input_shape=(6,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='elu', name='elu'))

    model.add(Dense(2048, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='leaky_relu', name='relu1'))
    model.add(Dropout(0.3))

    model.add(Dense(4096, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.000001)))
    model.add(Activation(activation='leaky_relu', name='relu2'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, name='fc4', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.000001))) #32
    model.add(Activation(activation='leaky_relu', name='relu3'))
    model.add(Dropout(0.3))

    model.add(Dense(128, name='fc6', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001))) #32
    model.add(Activation(activation='elu', name='elu2'))

    #hls_synth_success
    model.add(Dense(1, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)))
    return model

def create_model_regression():
    model = Sequential()
    # d_in	d_out prec rf strategy (one-hot encoded)
    model.add(Dense(64, input_shape=(7,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='leaky_relu', name='relu1'))
    #model.add(BatchNormalization())
    model.add(Dense(1024, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='leaky_relu', name='relu2'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001)))
    model.add(Activation(activation='elu', name='elu1'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(1024, name='fc4', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001))) #32
    model.add(Activation(activation='leaky_relu', name='relu3'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, name='fc6', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.00001))) #32
    model.add(Activation(activation='elu', name='elu2'))
    #hls_synth_success, WorstLatency_hls, IntervalMax_hls, FF_hls, LUT_hls, BRAM_18K_hls, DSP_hls
    model.add(Dense(6, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.001)))
    return model
