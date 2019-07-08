import numpy as np
import utils_tf
import utils_np
import models
def base_graph(state_value, atom_num, act, dimension, nodes):
    """Define an initial graph structure for the initial state.

    A state is represented by a set of atoms. And an action a operates on the set of the grounded atoms, making
    the current state becomes the next one.

    Args:
      state_value: a vector recording the atoms of the states
      atom_num: the size of all the atoms
      act: the input node for the action
      dimension: the dimension of the node & edge features

    Returns:
      data_dict: dictionary with nodes, edges, receivers and senders
          to represent a structure like the one above.
    """
    # Nodes
    # Nodes[0] -> state
    # Nodes[atom_num+1] -> action

    edges, senders, receivers = [], [], []

    for i in range(0, atom_num + 1):
        if i == 0:
            for vindex in range(1, atom_num + 1):
                if state_value[vindex - 1] == 1:
                    edges.append([1])
                else:
                    edges.append([0])
                senders.append(vindex)
                receivers.append(0)

    edges = np.array(edges, dtype='float32')
    return {
        "globals": act,
        "nodes": nodes,
        "edges": edges,
        "receivers": receivers,
        "senders": senders,
    }


def make_all_runnable_in_session(*args):
    """Apply make_runnable_in_session to an iterable of graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]

