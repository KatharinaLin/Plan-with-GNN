import modules
import utils_tf
import blocks
import numpy as np
import tensorflow as tf
import sonnet as snt

def make_rnn_model():
    self.hidden_size = 20
    valid_regularizers = {
        "in_to_hidden": {
            "w": tf.nn.l2_loss,
        },
        "hidden_to_hidden": {
            "b": tf.nn.l2_loss,
        }
    }
    return snt.Sequential([
        snt.VanillaRNN(name="rnn",hidden_size=self.hidden_size,egularizers=valid_regularizers),
                           snt.LayerNorm()
    ])

def make_mlp_model():
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  regularizers = {"w": tf.contrib.layers.l1_regularizer(scale=0.5),
                  "b": tf.contrib.layers.l2_regularizer(scale=0.5)}
  return snt.Sequential([
      snt.nets.MLP([100,100,100,100], activate_final=True),
      snt.LayerNorm()
  ])

class Encoder(snt.AbstractModule):
  """Encoder of the states"""

  def __init__(self, name="Encoder"):
    super(Encoder, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = make_mlp_model()

  def _build(self, inputs):
    return self._network(inputs)

class Decoder(snt.AbstractModule):
  """Encoder of the states"""

  def __init__(self, name="Encoder"):
    super(Decoder, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = make_mlp_model()

  def _build(self, inputs):
    return self._network(inputs)

class SigmoidProcessingModule(snt.AbstractModule):
  def __init__(self, output_size, name="sigmoid_output_func"):
    super(SigmoidProcessingModule, self).__init__(name=name)
    self._output_size = output_size
    with self._enter_variable_scope():  # This line is crucial!
      self._lin_mod = snt.Linear(self._output_size, name="state_edge_output")
  def _build(self, inputs):
      return tf.nn.sigmoid(self._lin_mod(inputs))

class LearningModule():
    def __init__(self, output_size,name="Learning"):
        super(HeuristicNetwork, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = make_heu_mlp_model()
        self._encoder = Encoder()
        self._decoder = Decoder()
        self._rnn = make_rnn_model()
        self.output_state_fn = lambda: SigmoidProcessingModule(output_size)
    def _build(self, state_inputs, action_inputs, num_processing_steps):
        en_state_inputs = self._encoder(state_inputs)
        prev_states = en_state_inputs
        de_prev_states = self._decoder(prev_states)
        output_state = self.output_state_fn(de_prev_states)
        output_states = []
        output_states.append(output_state)
        for i in range(num_processing_steps):
            next_states = self._rnn(prev_states, action_inputs[i])
            de_next_states = self._decoder(next_states)
            output_state = self.output_state_fn(de_next_states)
            prev_states = next_states
            output_states.append(output_state)
        return output_states