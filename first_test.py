

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import sys
import numpy as np

import utils_tf
import utils_np
import models
from matplotlib import pyplot as plt
from prev import ParseFile as PF
from models import HeuristicNetwork
import numpy as np
import sonnet as snt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import sys
import csv


domain_names = ["Log","Depot","Zeno"]
domain_name = domain_names[int(sys.argv[1])]
data_number = sys.argv[2]
percent_number = sys.argv[3]
dataset = int(data_number)
save_per = sys.argv[4]
# domain_name = domain_names[0]
# data_number = "100"
# percent_number = "1.0"
# dataset = 100
# save_per = "100"
training_heu_times = 6000
testing_steps=10
halt_steps = 50
batch_size_tr = 50
# Data / training parameters.
num_training_iterations = 20000



os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[5]
try:
    import seaborn as sns
except ImportError:
    pass
else:
    sns.reset_orig()

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)
halt_steps = 50

# @title Helper functions  { form-width: "30%" }

# pylint: disable=redefined-outer-name

def base_graph(state_value, atom_num, act, dimension):
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
    nodes = (np.random.uniform(-6 / np.sqrt(dimension), 6 / np.sqrt(dimension), (atom_num + 1) * dimension))
    nodes = [1] * ((atom_num + 1) * dimension)
    nodes = np.reshape(nodes, (atom_num + 1,dimension))

    nodes = np.array(nodes, dtype='float32')
    edges, senders, receivers = [], [], []

    edge_count = 0
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


def computeRelativity(edges, i, j):
    e_mean = tf.reduce_mean(edges, axis=0)
    rij_frac_1 = 0
    n = len(edges)
    i_sum = 0
    j_sum = 0
    for k in range(0, n):
        rij_frac_1 += ((edges[k][i, :] - e_mean[i, :]) * (edges[k][j, :] - e_mean[j, :]))
        i_sum += ((edges[k][i, :] - e_mean[i, :]) ** 2)
        j_sum += ((edges[k][j, :] - e_mean[j, :]) ** 2)

    rij = rij_frac_1 / (tf.sqrt(i_sum) * tf.sqrt(j_sum))
    # rij = a / (tf.sqrt(b) * tf.sqrt(c))
    return rij


def create_loss_ops(all_output_ops, atom_num, finals, hidden):
    """Create loss function

    Args:
      output_op: The last output graphs from the model.

    Returns:
      A list of loss values (tf.Tensor), one per output op.
    """


    batch_size = len(all_output_ops)
    all_losses = []
    for i in range(batch_size):
        output_ops = all_output_ops[i]
        final = finals[i]
        loss_ops = []
        index = -1

        count = 0
        for output_op in output_ops:
            count += 1
            if index == -1:
                index += 1
                continue
            atom_state_edges = output_op.edges[0:atom_num, ...]
            final_state = final[index, :, :]
            index += 1
            temp = tf.subtract(1.0, final_state)
            tmp = tf.log(tf.clip_by_value(tf.subtract(1.0, atom_state_edges),1e-8,1))

            loss1 = - tf.multiply(final_state, tf.log(tf.clip_by_value(atom_state_edges,1e-8,1))) - tf.multiply(temp, tmp)
            if count != 1 and count != len(output_ops):
                tmp2 = hidden[i][index-1]
                loss1 = tf.multiply(loss1, tmp2)

            loss1sum = tf.reduce_mean(loss1)
            all_losses.append(loss1sum)

    # all_losses.append(temp)
    return tf.reduce_mean(all_losses)


def create_heu_loss(output,action):
    temp = tf.reduce_mean((output - action)**2)
    return temp



def getRelatedSet(valued3_state):
    count = 0
    initial = []
    for element in valued3_state:
        if element == 1:
            initial.append(count)
            count += 1
    return initial


def getFinalState(state_batch, num_processing_steps):
    finals = []
    for allState in state_batch:
        allState = allState[1: num_processing_steps + 1]
        final_array = []
        for final in allState:
            final_array_ele = []
            for element in final:
                if element == -1:
                    final_array_ele.append([0.0])
                else:
                    final_array_ele.append([1.0])
            final_array.append(final_array_ele)
        finals.append(final_array)
    return finals


def error_calculation(output_ops_test, atom_num, states):

    loss_op_test = []
    error_array = []
    for output_op_test, state in zip(output_ops_test, states):
        index_o = 0
        errors = []
        for st in state:
            output_ops = output_op_test[index_o]
            count = 0
            for index in range(len(output_ops)):
                tmp = -1.0
                ele = output_ops[index]
                if ele >= 0.5:
                    tmp = 1.0
                if tmp != st[index]:
                    count += 1
            errors.append(count)
            index_o += 1
        counts = sum(errors) / index_o
        error_array.append(counts)

    return sum(error_array) / len(output_ops_test)

def TrainDataAcquire(all_output_tr, all_action_set):
    training_sample_input = []
    training_sample_output = []
    allvsg = []
    for output_trs, action_set in zip(all_output_tr, all_action_set):
        combination = []
        for i in range(len(output_trs)-1):
            for j in range(i+1, len(output_trs)):
                combination.append(output_trs[i].nodes[0])
                combination.append(output_trs[j].nodes[0])
                combination = tf.concat(combination, axis=-1)
                training_sample_input.append(tf.stack([combination]))
                training_sample_output.append(action_set[i])
                combination = []
        allvsg.append(output_trs[len(output_trs)-1].nodes[0])
    training_sample_input = tf.concat(training_sample_input, axis=0)
    training_sample_output = tf.concat(training_sample_output, axis=0)
    return training_sample_input, training_sample_output, allvsg



def getActionIndex(act_vec, output_vecs, action_set_size):
    minIndex = 0
    minDiff = tf.reduce_sum((act_vec-output_vecs[0])**2)
    for i in range(action_set_size):
        minDiff =tf.cond(tf.reduce_sum((act_vec-output_vecs[i])**2) < minDiff, lambda :tf.reduce_sum((act_vec-output_vecs[0])**2), lambda :minDiff)
        minIndex = tf.cond(tf.reduce_sum((act_vec-output_vecs[i])**2) < minDiff, lambda : i, lambda : minIndex)
    return minIndex


def getStopIndex(planning_output, endState):
    index = 0
    end_list = [[0]] * len(endState)
    for arr_ele in endState:
        if arr_ele == 1:
            end_list[index] = [1]
        index += 1
    index = 0
    end_list = np.array(end_list)
    for intermit in planning_output:
        if sum(intermit-end_list)[0] == 0:
            break
        index += 1
    return index
def state_data_processing(v3_state, num_processing_steps):
    while len(v3_state) < num_processing_steps+1:
        v3_state=np.row_stack((v3_state,v3_state[len(v3_state)-1]))
    return v3_state

def action_index_sequence_processing(acts, acts_test, leng, num_processing_steps):
    for act in acts:
        while len(act) < num_processing_steps:
            act.append(leng)
    for act in acts_test:
        while len(act) < num_processing_steps:
            act.append(leng)
    return acts, acts_test

tf.reset_default_graph()

rand = np.random.RandomState(SEED)


batch_size_tr = 10




dimension = 3
state = []
action_input = []
grounding = []
count = 0
action_dict = {}

num_processing_steps = 15
acts =[]
hidden_bit = []
for k in range(1, 1+dataset):
    filepath = "./prev/solution/"+domain_name+"/"+domain_name+percent_number+"/"+domain_name+"_solution/input_" + str(k) + ".txt"
    _, _, groundings, actions, valued3_state, hidden_arr, _ = PF.parsefile(filepath)

    valued3_state = state_data_processing(valued3_state, num_processing_steps)
    state.append(valued3_state)
    action_seq = []
    for act in actions:
        if action_dict.__contains__(act):
            action_seq.append(action_dict[act])
        else:
            action_dict[act] = count
            count += 1
            action_seq.append(action_dict[act])

    hidden = []
    for arr_ele in hidden_arr:
        temp = [[0]] * len(groundings)
        for ele in arr_ele:
            temp[ele] = [1]
        hidden.append(temp)
    while len(hidden) < num_processing_steps:
        hidden.append([[1]] * len(groundings))


    acts.append(action_seq)
    hidden_bit.append(hidden)
    grounding = groundings


# state, state_test, acts, acts_test = train_test_split(state, acts, test_size=0.2)
acts_test = []
state_test = []
count = 0
state_test_real_goal = []
for k in range(dataset+1, dataset+26):
    filepath = "./prev/solution/"+domain_name+"/"+domain_name+percent_number+"/"+domain_name+"_solution/input_" + str(k) + ".txt"
    _, _, groundings, actions, valued3_state, _, state_test_real_goal = PF.parsefile(filepath)

    valued3_state = state_data_processing(valued3_state, num_processing_steps)
    state_test.append(valued3_state)
    action_seq = []
    for act in actions:
        if action_dict.__contains__(act):
            action_seq.append(action_dict[act])
        else:
            action_dict[act] = count
            count += 1
            action_seq.append(action_dict[act])


    acts_test.append(action_seq)

acts, acts_test = action_index_sequence_processing(acts, acts_test, len(action_dict), num_processing_steps)










first_action = np.random.uniform(-6 / np.sqrt(dimension), 6 / np.sqrt(dimension), dimension)
first_action = np.array(first_action, dtype='float32')

base_graph_set = []
base_graph_set_test = []
for v3_state in state:
    static_graph_tr = [base_graph(v3_state[0], len(grounding), first_action, dimension)]
    base_graph_tr = utils_tf.data_dicts_to_graphs_tuple(static_graph_tr)
    base_graph_set.append(base_graph_tr)
    base_graph_tr = make_all_runnable_in_session(base_graph_tr)

goal_graph_test = []
for v3_state in state_test:
    static_graph_test = [base_graph(v3_state[0], len(grounding), first_action, dimension)]
    goal_graph = [base_graph(v3_state[num_processing_steps], len(grounding), first_action, dimension)]
    base_graph_test = utils_tf.data_dicts_to_graphs_tuple(static_graph_test)
    base_graph_set_test.append(base_graph_test)
    temp = utils_tf.data_dicts_to_graphs_tuple(goal_graph)
    goal_graph_test.append(temp)
    base_graph_test = make_all_runnable_in_session(base_graph_test)


atom_num = len(grounding)
number = (atom_num + 2) * atom_num


graph_index = tf.placeholder(tf.int32, shape=(None,))
final = tf.placeholder(tf.float32, shape=([None,num_processing_steps,atom_num, 1]))
graph_index_test = tf.placeholder(tf.int32, shape=(None,))
final_test = tf.placeholder(tf.float32, shape=([1,num_processing_steps,atom_num, 1]))
action_seqs = tf.placeholder(tf.int32, shape=([None, num_processing_steps]))
action_seqs_test = tf.placeholder(tf.int32, shape=([num_processing_steps]))
hiddens = tf.placeholder(tf.float32, shape=([None,num_processing_steps, atom_num, 1]))

model = models.GraphProcess(edge_output_size=1, dimension=dimension,
                            atom_num=atom_num, action_set_size=len(action_dict)+1)


restore_action = []
with open(domain_name+"-n"+data_number+"-p"+save_per+"_action.csv", newline='') as csvfile:
    action_file = csv.reader(csvfile)
    for data in action_file:
        re_action = []
        for eachone in data:
            re_action.append(float(eachone))
        restore_action.append(re_action)

restore_pn = []
with open(domain_name+"-n"+data_number+"-p"+save_per+"_pnode.csv", newline='') as csvfile:
    pnode_file = csv.reader(csvfile)
    for data in pnode_file:
        re_pn = []
        for eachone in data:
            re_pn.append(float(eachone))
        restore_pn.append(re_pn)

model.setActionStore(restore_action)
model.setPNode(restore_pn)



all_output_ops_tr = []
all_acts = []
Indices = []
for i in range(len(state)):
    Indices.append(i)


all_state_vecs = []
for j in range(batch_size_tr):
    graph = base_graph_set[0]
    for i in range(0, len(base_graph_set)):
        graph = tf.cond(tf.less_equal(i,graph_index[j]), lambda: base_graph_set[i], lambda: graph)

    output_ops_tr, action_set, sta_vecs = model(graph, action_seqs[j], num_processing_steps, True)
    all_output_ops_tr.append(output_ops_tr)
    all_state_vecs.append(sta_vecs)
    all_acts.append(action_set)









# Testing
output_ops_test = []
output_model = []
graph_test = base_graph_set_test[0]
goal_graph_graph_test = goal_graph_test[0]

for i in range(0, len(base_graph_set_test)):
    graph_test = tf.cond(tf.less_equal(i,graph_index_test[0]), lambda: base_graph_set_test[i], lambda: graph_test)
    goal_graph_graph_test = tf.cond(tf.less_equal(i,graph_index_test[0]), lambda: goal_graph_test[i], lambda: goal_graph_graph_test)

output_op_test, _,sta_vec = model(graph_test, action_seqs_test, num_processing_steps, False)
output_model = output_ops_test
output_ops_test= []
for o_o_t in output_op_test:
    output_ops_test.append(o_o_t.edges)


# Training Loss

loss_ops_trs = []
loss_ops_tr = create_loss_ops(all_output_ops_tr, atom_num, final, hiddens)


loss_op_tr = tf.reduce_sum(loss_ops_tr)
# Optimizer.
params = tf.trainable_variables()
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)





output_action = model.getActionStore()
origin_action = model.getOriginActionStore()
p_node = model.getPNode()



init_op = tf.global_variables_initializer()

# saver = tf.train.Saver()

restore_var = tf.trainable_variables()[0:42]
saver = tf.train.Saver(restore_var)

restore_vars = tf.trainable_variables()


# Runtime
test_result = []
x_axis = []
losses = []
Indices_test = []
sample_size_test = len(state_test)
sample_size = len(acts)
for i in range(len(state_test)):
    Indices_test.append(i)

with tf.Session() as sess:

    sess.run(init_op)
    saver.restore(sess, "./model/GNN-model-"+domain_name+"-n"+data_number+"-p"+save_per+".ckpt")
    last_iteration = 0
    log_every_seconds = 20

    start_time = time.time()
    last_log_time = start_time
    losses_tr = []
    logged_iterations = []
    
    count = 0
    flags = [0] * sample_size
    index = 0

    train_values = []


    flagstop = 0


    test_array = []
    for iteration in range(0, len(state_test)):
        # Training feed_dict
        startIndex = (iteration * batch_size_tr) % sample_size
        g_index = []
        state_batch = []
        act_indexs = []
        hidden_batch = []

        if startIndex + batch_size_tr <= sample_size:
            endIndex = startIndex + batch_size_tr
            g_index = Indices[startIndex:endIndex]
            state_batch = getFinalState(state[startIndex:endIndex], num_processing_steps)
            act_indexs = acts[startIndex:endIndex]
            hidden_batch = hidden_bit[startIndex:endIndex]
        else:
            endIndex = (startIndex + batch_size_tr) % sample_size
            g_index = Indices[startIndex:sample_size]
            act_indexs = acts[startIndex:sample_size]
            hidden_batch = hidden_bit[startIndex:sample_size]
            state_batch = state[startIndex:sample_size]
            if endIndex != 0:
                temp = Indices[0:endIndex]
                temp1 = acts[0:endIndex]
                temp2 = state[0:endIndex]
                temp3 = hidden_bit[0:endIndex]
            else:
                temp = Indices
                temp1 = acts
                temp2 = state
                temp3 = hidden_bit
            g_index += temp
            act_indexs += temp1
            hidden_batch += temp3
            state_batch += temp2
            state_batch = getFinalState(state_batch, num_processing_steps)
        act_indexs = np.array(act_indexs)
        state_batch = np.array(state_batch)
        g_index = np.array(g_index)
        hidden_batch = np.array(hidden_batch)

        # Test feed_dict

        act_indexs_test = []
        state_batch_test = []
        start_index_test = iteration % sample_size_test
        act_indexs_test = np.array(acts_test[start_index_test])
        act_indexs_test.reshape((1, len(act_indexs_test)))
        state_batch_test = getFinalState([state_test[start_index_test]], num_processing_steps)
        g_index_test = [Indices_test[start_index_test]]

        last_iteration = iteration
        train_values = sess.run({
            "test_ops": output_ops_test,
            "out":output_action,
            "origin":origin_action,
            "p_node":p_node,
        }, feed_dict={graph_index: g_index, final: state_batch, action_seqs: act_indexs, graph_index_test: g_index_test,
                      final_test: state_batch_test, action_seqs_test: act_indexs_test, hiddens: hidden_batch})

        test_array.append(train_values['test_ops'])

    tmp = error_calculation(test_array, atom_num, state_test)
    print("Test {:4f}".format(
            tmp))











