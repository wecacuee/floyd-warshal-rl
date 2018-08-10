import copy
import functools
import operator
# Domain specific language for defining graph
# Should not depend on third party modules
# Implementation is handled by *driver*
class CompGraph(object):
    _counter = 0
    """ The function to call to create a corresponding graph element """
    _driver_mapping = {}
    def __init__(self, *driver_args, **kwargs):
        # For multi output graph
        self._noutputs = kwargs.pop("compgraph.outputs", 1)
        self.inputs = None
        # arguments to driver
        self.driver_kwargs = kwargs
        self.driver_args = driver_args
        # Container to collect variables
        # Variables are state variables that considered the part of the graph
        # and usually not inputted from outside
        self.variables = []
        # Underlying driver graph
        self._driver_graph = None
        # Counter kept for naming layers and variables
        self._my_id = self._counter
        self.__class__._counter += 1

    def __call__(self, *inputs):
        self.inputs = inputs
        return self

    def outputs(self):
        if self._noutputs == 1:
            return self
        else:
            return [_CGWrap(self, i) for i in range(self._noutputs)]

    def decendent_variables(self):
        """ Return all variables including decendent's variables """
        return self.variables + reduce(
            lambda v, i: v + i.decendent_variables()
            , self.inputs, [])

    def replace_driver_kwargs(self, **kwargs):
        for k in [k for k in kwargs.keys()
                  if k in self.driver_kwargs]:
            if self.driver_kwargs[v] is None:
                self.driver_kwargs[k] = kwargs[k]

        for i in self.inputs:
            i.replace_driver_kwargs(kwargs)

    def name(self):
        """ Create a unique name for each object"""
        if self._my_id:
            return "%s_%d" % (self.type, self._my_id)
        else:
            return self.type

    @property
    def type(self):
        return self.__class__.__name__

    @staticmethod
    def issubclass_safe(x, P):
        return isinstance(x, type) and issubclass(x, P)

    def get_driver_mapping(self, attr):
        return (attr.driver_graph()
                if isinstance(attr, CompGraph)
                else attr)

    def _build_driver_graph(self):
        driver_op = self.driver_mapping[self.type]
        if self.inputs is not None:
            _args = [ch.driver_graph() for ch in self.inputs
                    ] + list(self.driver_args)
        else:
            _args = self.driver_args
        _kwargs = dict(
            (k, self.get_driver_mapping(v))
            for k, v in self.driver_kwargs.items())
        # Return a partial function if inputs are not set
        if self.inputs is None:
            self._driver_graph = functools.partial(driver_op
                                                   , self
                                                   , *_args
                                                   , **_kwargs)
        else:
            self._driver_graph = driver_op( self , *_args , **_kwargs)

    def driver_graph(self):
        if self._driver_graph is None:
            self._build_driver_graph()
        return self._driver_graph

    @property
    def driver_mapping(self):
        return self._driver_mapping

    @driver_mapping.setter
    def driver_mapping(self, dm):
        CompGraph._driver_mapping = dm

class _CGWrap(CompGraph):
    def __init__(s, shared, driver_output_index):
        s.shared = shared
        # For multi output graph
        s._driver_output_index = driver_output_index

    def getattr(s, a):
        return getattr(s.shared, a)

    def setattr(s, a, v):
        return setattr(s.shared, a, v)

    def driver_graph(s):
        return s.shared.driver_graph()[s._driver_output_index]


class LSTMCell(CompGraph):
    _P = CompGraph
    def __init__(self, *inputs, **driver_kwargs):
        kwargs = {"compgraph.outputs":2}
        kwargs.update(driver_kwargs)
        self._P.__init__(self, *inputs, **kwargs)

_comgraph = """placeholder """.split()
# UGLY but reduces typing
dslvocab = """placeholder xavier_initializer relu Variable Linear Conv2D
maximum exponential_decay one_hot truncated_normal_initializer reduce_max
reduce_mean reduce_sum """.split()

for v in dslvocab:
    globals()[v] = type(v, (CompGraph,), dict())

dslvocab.append(LSTMCell.__name__)


##### Tensor flow dependent code (Driver code) #############################


def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           initializer=lambda : tf.contrib.layers.xavier_initializer(),
           activation_fn=lambda x: tf.nn.relu(x),
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
  """
  Initializes shared variables for W, b and chains activation_fn(W (*) x + b)
  where (*) is convolution

  You better provide a meaning full name, otherwise you may end up
  unintentional sharing of weights
  """
  import tensorflow as tf
  with tf.variable_scope(name):
    if data_format == 'NCHW':
      stride = [1, 1, stride[0], stride[1]]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
    elif data_format == 'NHWC':
      stride = [1, stride[0], stride[1], 1]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

    w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
    conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

    b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(conv, b, data_format)

  if activation_fn != None:
    out = activation_fn(out)

  return out, w, b

def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))

    out = tf.nn.bias_add(tf.matmul(input_, w), b)

    if activation_fn != None:
      return activation_fn(out), w, b
    else:
      return out, w, b

def product(l_):
    return reduce(operator.mul, l_, 1)

def filter_kw(kw, names):
    return dict((k,v) for k, v in kw.items()
                if k in names)


class IgnOne():
    def __init__(self, foo):
        self.foo = foo

    def __call__(self, graph, *args, **kwargs):
        return self.foo(*args, **kwargs)

    def __str__(self):
        return str(self.foo)

class TFCompGraphDriver(object):
    @staticmethod
    def _conv2D(graph, input_, output_dim=None, kernel_size=None, stride=None,
                **kwargs):
        assert output_dim is not None
        assert kernel_size is not None
        assert stride is not None
        out, w, b = conv2d(input_ , output_dim , kernel_size , stride,
                           name=graph.name() ,  **kwargs)
        graph.variables.append(w)
        graph.variables.append(b)
        return out

    @staticmethod
    def _linear(graph, input_, output_size=None, **kwargs):
        # Outside the graph computation because we need the size of 
        # linear matrices right now before the Tensor graph is 
        # computed. If we depend upon the graph
        shape = input_.get_shape().as_list()
        if len(shape) > 2:
            new_shape = [-1, product(shape[1:])]
            input_2 = tf.reshape(input_,  new_shape)
        else:
            input_2 = input_

        out, w, b = linear(input_2, output_size, name=graph.name()
                           , **filter_kw(kwargs, "stddev".split()))
        graph.variables.append(w)
        graph.variables.append(b)
        return out

    @staticmethod
    def chained_gettattr(trials, attr):
        found = False
        out = None
        for t in trials:
            try:
                out = getattr(t, attr)
                found = True
                break
            except AttributeError:
                pass
        if not found:
            raise
        return out

    @staticmethod
    def _truncated_normal_initializer(graph, shape, **kwargs):
        init_kw = dict((k, kwargs.pop(k)) for k in kwargs.keys()
                       if k in "mean stddev seed dtype".split())
        return tf.truncated_normal_initializer(**init_kw)(shape, **kwargs)

    @classmethod
    def driver_mapping(cls):
        map = dict(
            Linear = cls._linear
            , Conv2D = cls._conv2D
            , reduce_sum = tf.reduce_sum
            , truncated_normal_initializer = \
              cls._truncated_normal_initializer
        )
        for v in dslvocab:
            if v not in map:
                foo = cls.chained_gettattr(
                    [tf, tf.nn, tf.nn.rnn_cell
                     , tf.train, tf.contrib.layers], v)
                #print(v, foo)
                map[v] = IgnOne(foo)
            else:
                pass
                #print(v, map[v])
        return map

def tfgraph(comp_graph):
    assert isinstance(comp_graph, CompGraph)
    comp_graph.driver_mapping = TFCompGraphDriver.driver_mapping()
    return comp_graph.driver_graph()

def make_DQN(action_size, input_shape, layer_scale = 32):
    initializer = truncated_normal_initializer(mean=0, stddev=0.02)
    activation_fn = relu()
    p1 = placeholder( dtype='float32' , shape=[None,] + list(input_shape))()
    p2 = Conv2D(
        output_dim = layer_scale
        , kernel_size = [8, 8]
        , stride = [4, 4]
        , initializer=initializer
        , activation_fn=activation_fn)(p1)
    p3 = Conv2D(
        output_dim = 2 * layer_scale
        , kernel_size = [4, 4]
        , stride = [2, 2]
        , initializer=initializer
        , activation_fn=activation_fn)(p2)
    p4 = Conv2D(
        output_dim = 2 * layer_scale
        , kernel_size = [3, 3]
        , stride = [1, 1]
        , initializer=initializer
        , activation_fn=activation_fn)(p3)
    p5 = Linear(output_size= 16 * layer_scale ,
                activation_fn=activation_fn)(p4)
    p6 = Linear(
        output_size=action_size
        , activation_fn=activation_fn)(p5)

    with tf.variable_scope('dqn'):
        assert isinstance(p6, CompGraph)
        return tfgraph(p6), tfgraph(p1)

def make_FC2_ops(input_, output_size, hidden_layer):
    activation_fn = tf.nn.relu
    fc2 = linear(
        linear(input_, hidden_layer, activation_fn=activation_fn
               , name="l1")[0]
        , output_size, activation_fn=activation_fn, name="l2")[0]
    return fc2

class FC2GRUCell():
    def __init__(self, output_size, hidden_layer, **kwargs):
        self.inherit_from = tf.nn.rnn_cell.RNNCell
        self._gru_cell = tf.nn.rnn_cell.GRUCell(**kwargs)
        self._output_size = output_size
        self._hidden_layer = hidden_layer
        self._second_call = False

    @property
    def state_size(self):
        return self._gru_cell.state_size

    @property
    def output_size(self):
        return self._gru_cell.output_size

    def __getattr__(self, attr):
        return getattr(self._gru_cell, attr)

    def __call__(self, input_, ph):
        with tf.variable_scope('gru') as scope:
            if self._second_call:
                scope.reuse_variables()
            self._second_call = True
            return self._gru_cell(
                make_FC2_ops(input_, self._output_size, self._hidden_layer)
                , ph)

def make_FC2RQN_ops(inputs, init_state, output_size, hidden_layer,
                    action_size, memory_size, stddev=0.02):
    with tf.variable_scope('fc2rqn') as scope:
        initializer = tf.truncated_normal_initializer(mean=0, stddev=stddev)
        output = []
        fc2gru_cell = FC2GRUCell( output_size = output_size
                                 , hidden_layer = hidden_layer
                                 , num_units=memory_size)
        cell_out = tf.nn.rnn_cell.OutputProjectionWrapper(fc2gru_cell
                                                       , action_size)
        outputs, out_state = tf.nn.dynamic_rnn(cell_out
                                       , inputs
                                       , initial_state = init_state
                                       , time_major=True)
        # ### Static
        # for s in range(max_steps):
        #     if s > 0:
        #         scope.reuse_variables()
        #     o, next_state = lstm_1(fc2, state)
        #     state = next_state
        #     output.append(o)
    return outputs, out_state


if __name__ == '__main__':
    with tf.Session() as sess:
        #dqn, input = make_DQN(5, (84, 84, 4))
        inputs = tf.placeholder('float32', shape=(10, None, 30))
        memory_size = 128
        init_state = tf.placeholder('float32', shape=(None, memory_size))
        outputs = make_FC2RQN_ops(
            inputs, init_state, output_size=128, hidden_layer=512, action_size=5
            , memory_size=memory_size)
        graph = outputs[0].graph
        filename = '/tmp/graphmodel-fc2rqn.protobuf'
        with open(filename, 'wb') as f :
            f.write(
                str(graph.as_graph_def()))
