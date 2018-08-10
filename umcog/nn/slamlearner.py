from __future__ import division, print_function

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 128

def make_lstm(vocab_size=256):
    import lasagne
    import theano.Tensor as T

    #Lasagne Seed for Reproducibility
    lasagne.random.set_rng(np.random.RandomState(1))
    l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size))
    l_in_process = lasagne.layers.DenseLayer(
        l_in, num_units=vocab_size, W = lasagne.init.Normal()
        , b=lasagne.init.Constant(0.1)
        , nonlinearity=lasagne.nonlinearities.rectify)
    l_lstm = lasagne.layers.LSTMLayer(
        l_in_process, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    l_out_process = lasagne.layers.DenseLayer(
        l_lstm, num_units=vocab_size, W= lasagne.init.Normal()
        , b=lasagne.init.Constant(0.1)
        , nonlinearity=lasagne.nonlinearities.rectify)
    network_output = lasagne.layers.get_output(l_out)
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    target_values = T.ivector('target_output')
    cost = T.nnet.categorical_crossentropy(network_output,
                                           target_values).mean()
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
