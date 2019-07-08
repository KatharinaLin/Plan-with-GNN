# @title Imports  { form-width: "30%" }

# The demo dependencies are not installed with the library, but you can install
# them with:
#
# $ pip install jupyter matplotlib scipy
#
# Run the demo with:
#
# $ jupyter notebook <path>/<to>/<demos>/shortest_path.ipynb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import sys
from numpy.core.multiarray import ndarray

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
import random
import csv


try:
    import seaborn as sns
except ImportError:
    pass
else:
    sns.reset_orig()



domain_names = ["Log","Depot","Zeno"]
# domain_name = domain_names[0]
# data_number = "100"
# percent_number = "1.0"

# dataset = 100
# save_per = "100"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
domain_name = domain_names[int(sys.argv[1])]
data_number = sys.argv[2]
percent_number = sys.argv[3]
dataset = int(data_number)
save_per = sys.argv[4]
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[5]
training_heu_times = 30000
typea="plan"
halt_steps = 15
testing_steps = 15



SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)


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
            tmp = tf.log(tf.subtract(1.0, atom_state_edges))

            loss1 = - tf.multiply(final_state, tf.log(atom_state_edges)) - tf.multiply(temp, tmp)
            if count != 1 and count != len(output_ops):
                loss1 = tf.multiply(loss1, hidden[index])

            loss1sum = tf.reduce_mean(loss1)
            all_losses.append(loss1sum)

    # all_losses.append(temp)
    return tf.reduce_mean(all_losses)


def create_heu_loss(output,action):
    # temp = tf.reduce_mean(((output - action)**2))
    temp = tf.subtract(1.0, action)
    tmp = tf.log(tf.clip_by_value(tf.subtract(1.0, output),1e-8,1))

    loss1 = - tf.multiply(action, tf.log(tf.clip_by_value(output,1e-8,1))) - tf.multiply(temp, tmp)
    loss1sum = tf.reduce_mean(loss1)
    return tf.reduce_sum(loss1sum)






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
        output_ops = output_op_test[0:atom_num, ...]
        count = 0
        for index in range(len(output_ops)):
            tmp = -1.0
            ele = output_ops[index]
            if ele >= 0.5:
                tmp = 1.0
            if tmp != state[num_processing_steps][index]:
                count += 1
        error_array.append(count)

    return sum(error_array) / len(output_ops_test)

def TrainDataAcquire(all_output_tr, all_action_set, final_state_vec, flagsNumber):
    training_sample_input = []
    training_sample_output = []
    allvsg = []
    index = 0
    for output_trs, action_set, final_state, fn in zip(all_output_tr, all_action_set, final_state_vec, flagsNumber):
        combination = []
        fn = tf.cast(fn, tf.int32)
        input_state_vec = output_trs[0].nodes[0]
        for i in range(len(output_trs)-1):

            input_state_vec = tf.cond(tf.greater(i+1, fn), lambda: input_state_vec, lambda : output_trs[i].nodes[0])

            combination.append(input_state_vec)
            combination.append(final_state)
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


def check_stop(next, real):
    check_num = len(real)
    count = 0
    for real_goal in real:
        if next[real_goal][0] == 1:
            count += 1
    if count == check_num:
        return True
    return False

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
dimension=3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
state = []
action_input = []
grounding = []
count = 0
action_dict = {}

num_processing_steps = 15
acts =[]
hidden_bit = []
for k in range(1, dataset+1):
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

for k in range(1+dataset, dataset+26):
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
goal_graph_train = []
for v3_state in state:
    static_graph_tr = [base_graph(v3_state[0], len(grounding), first_action, dimension)]
    goal_graph = [base_graph(v3_state[testing_steps], len(grounding), first_action, dimension)]
    base_graph_tr = utils_tf.data_dicts_to_graphs_tuple(static_graph_tr)
    base_graph_set.append(base_graph_tr)
    base_graph_tr = make_all_runnable_in_session(base_graph_tr)
    temp = utils_tf.data_dicts_to_graphs_tuple(goal_graph)
    goal_graph_train.append(temp)

goal_graph_test = []
test_edges = []
for v3_state in state_test:
    static_graph_test = [base_graph(v3_state[0], len(grounding), first_action, dimension)]
    goal_graph = [base_graph(v3_state[testing_steps], len(grounding), first_action, dimension)]
    base_graph_test = utils_tf.data_dicts_to_graphs_tuple(static_graph_test)
    base_graph_set_test.append(base_graph_test)
    test_edges.append(base_graph_test.edges)
    temp = utils_tf.data_dicts_to_graphs_tuple(goal_graph)
    goal_graph_test.append(temp)
    # base_graph_test = make_all_runnable_in_session(base_graph_test)


atom_num = len(grounding)
number = (atom_num + 2) * atom_num


graph_index = tf.placeholder(tf.int32, shape=(None,))
final = tf.placeholder(tf.float32, shape=([None,num_processing_steps,atom_num, 1]))
graph_index_test = tf.placeholder(tf.int32, shape=(None,))
final_test = tf.placeholder(tf.float32, shape=([1,num_processing_steps,atom_num, 1]))
action_seqs = tf.placeholder(tf.int32, shape=([None, num_processing_steps]))
action_seqs_test = tf.placeholder(tf.int32, shape=([num_processing_steps]))
hidden = tf.placeholder(tf.float32, shape=([None,num_processing_steps, atom_num, 1]))
current_edges = tf.placeholder(tf.float32, shape=([atom_num, 1]))
maxIndex_chosen_index = tf.placeholder(tf.int32,shape=())
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
final_state_vecs = []
for j in range(batch_size_tr):
    graph = base_graph_set[0]
    goal_graph_graph_train = goal_graph_train[0]
    for i in range(0, len(base_graph_set)):
        graph = tf.cond(tf.less_equal(i,graph_index[j]), lambda: base_graph_set[i], lambda: graph)
        goal_graph_graph_train = tf.cond(tf.less_equal(i,graph_index[j]), lambda: goal_graph_train[i], lambda: goal_graph_graph_train)
    output_ops_tr, action_set, sta_vecs = model(graph, action_seqs[j], num_processing_steps, False)
    vsg_train = model.getInitialVecForState(goal_graph_graph_train)
    vsg_train = vsg_train.nodes[0]
    final_state_vecs.append(vsg_train)
    all_output_ops_tr.append(output_ops_tr)
    all_state_vecs.append(sta_vecs)
    all_acts.append(action_set)


output_action = model.getActionStore()
action_size = len(action_dict)
onehot = []
flags_number = []
for i in range(batch_size_tr):
    each_batch = action_seqs[i]
    temp = []
    index = each_batch[0]
    tmp2 = tf.one_hot(indices=[index], depth=action_size+1)
    prev_tmp = tmp2
    for j in range(num_processing_steps):
        index = each_batch[j]
        tmp = tf.one_hot(indices=[index], depth=action_size+1)
        tmp2 = tf.cond(tf.equal(index, len(action_dict)), lambda: prev_tmp, lambda: tmp)
        prev_tmp = tmp2
        temp.append(tmp2)
    argMaxIn = tf.argmax(each_batch)
    flags_number.append(argMaxIn)
    onehot.append(temp)

training_sample_input, training_sample_output, allvsg = TrainDataAcquire(all_state_vecs, onehot, final_state_vecs, flags_number)
heuristic = HeuristicNetwork(name="heuristic", size=len(action_dict)+1)
heuristic_output = heuristic(training_sample_input)









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
output_ops_test.append(output_op_test[len(output_op_test) - 1].edges)


# Training Loss

loss_ops_trs = []
loss_ops_tr = create_loss_ops(all_output_ops_tr, atom_num, final, hidden)
loss_heu = create_heu_loss(heuristic_output,training_sample_output)

loss_op_tr = tf.reduce_sum(loss_ops_tr)
# Optimizer.
params = tf.trainable_variables()
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)
optimizers = tf.train.AdamOptimizer(learning_rate)
param_list = tf.trainable_variables()[42:]
step_heu = optimizers.compute_gradients(loss_heu, param_list)
update_para = optimizers.apply_gradients(step_heu)
p_list =  tf.trainable_variables()
i_count = 0


#planning
ans = []
act_vecs = []

planning_output = []
output_action = model.getActionStore()
plans = []
output_trans = 0
vsg = model.getInitialVecForState(goal_graph_graph_test)
vsg = vsg.nodes[0]
goal_edges = goal_graph_graph_test.edges
count = 0
vecs = []
latent = []
graph_test = graph_test.replace(edges=current_edges)


output_trans = model.getInitialVecForState(graph_test)
vs0 = output_trans.nodes[0]


concat_array = []
concat_array.append(vs0)
concat_array.append(vsg)
HeuristicInput = tf.concat(concat_array, axis=-1)
HeuristicInput = tf.stack([HeuristicInput])
act_vec = heuristic(HeuristicInput)



wait_action = 0
wait_action_number = testing_steps
wait_index = []

maxIndex=(tf.argmax(act_vec, axis=-1))[0]
temp = act_vec
while wait_action < wait_action_number:
    wait_index_tmp = (tf.argmax(temp,axis=-1))[0]
    wait_index.append(wait_index_tmp)
    tmp = tf.one_hot(wait_index_tmp,depth=len(action_dict)+1, on_value=0,off_value=1)
    tmp = tf.cast(tmp, tf.float32)
    temp = tf.multiply(temp, tmp)
    wait_action += 1

wait_index = tf.convert_to_tensor(wait_index)
# maxIndex = wait_index[tf.random_uniform(shape=[],minval=0,maxval=wait_action_number-1,dtype=tf.int32)]
maxIndex = wait_index[maxIndex_chosen_index]

latent, output_trans = model.processOneStep(output_trans, maxIndex)

next_edges = output_trans.edges










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





    last_iteration = 0
    
    
    for iteration in range(last_iteration, training_heu_times):
    
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
            state_batch += temp2
            hidden_batch += temp3
            state_batch = getFinalState(state_batch, num_processing_steps)
        act_indexs = np.array(act_indexs)
        state_batch = np.array(state_batch)
        g_index = np.array(g_index)
        hidden_batch = np.array(hidden_batch)

        cur_edge = np.array([[1]]*atom_num)
        # Test feed_dict
        acts_test = np.array(acts_test)
        act_indexs_test = []
        state_batch_test = []
        start_index_test = iteration % sample_size_test 
        act_indexs_test = np.array(acts_test[start_index_test])
        act_indexs_test.reshape((1, len(act_indexs_test)))
        state_batch_test = getFinalState([state_test[start_index_test]], num_processing_steps)
        g_index_test = [Indices_test[start_index_test]]
        hidden_batch = np.array(hidden_batch)
        last_iteration = iteration
        train_values = sess.run({
            "step": update_para,
            "loss": loss_heu,
            "heu":heuristic_output,
            "out":training_sample_output,
            "input":training_sample_input,
            "out1`":all_output_ops_tr,
            "state":all_state_vecs,
            "action":output_action,
            "all_state_vecs":all_state_vecs,
            "graph":graph,
            "action_seqs":action_seqs,
            "test_edges":test_edges,
            "output_action":output_action,
            "Msx":flags_number,
        }, feed_dict={graph_index: g_index, final: state_batch, action_seqs: act_indexs, graph_index_test: g_index_test,
                      final_test: state_batch_test, action_seqs_test: act_indexs_test, hidden:hidden_batch, current_edges:cur_edge, maxIndex_chosen_index:0})
        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        last_log_time = the_time
        elapsed = time.time() - start_time
        losses_tr.append(train_values["loss"])
        logged_iterations.append(iteration)
        print("# {:05d}, T {:.1f}, Ltr {:6f}".format(iteration, elapsed, train_values["loss"]))
        if train_values["loss"] < 0.0000001:
            break
    test_array = []
    plan_array = []
    for iteration in range(0, len(state_test)):
        cur_edge = train_values['test_edges'][iteration]
        planning_act_seq = []
        maxIndex_chosen_index_defined = 0
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
            state_batch += temp2
            hidden_batch += temp3
            state_batch = getFinalState(state_batch, num_processing_steps)
        act_indexs = np.array(act_indexs)
        state_batch = np.array(state_batch)
        g_index = np.array(g_index)

        # Test feed dict
        act_indexs_test = []
        state_batch_test = []
        start_index_test = iteration % sample_size_test
        act_indexs_test = np.array(acts_test[start_index_test])
        act_indexs_test.reshape((1, len(act_indexs_test)))
        state_batch_test = getFinalState([state_test[start_index_test]], num_processing_steps)
        g_index_test = [Indices_test[start_index_test]]

        last_iteration = iteration
        action_num = 0
        while action_num < halt_steps:


            train_value = sess.run({
                "next": next_edges,
                "ans": maxIndex,
                "cur":current_edges,
                "wait_index":wait_index,
                "act_vec":act_vec,
            }, feed_dict={graph_index: g_index, final: state_batch, action_seqs: act_indexs,
                          graph_index_test: g_index_test,
                          final_test: state_batch_test, action_seqs_test: act_indexs_test, hidden: hidden_batch, current_edges:cur_edge, maxIndex_chosen_index:maxIndex_chosen_index_defined})

            if train_value["ans"] in planning_act_seq:

                # action can not be repeated in 5 actions
                if len(planning_act_seq) < 5:
                    cur_edge = train_value["cur"]
                    maxIndex_chosen_index_defined += 1

                else:
                    if train_value["ans"] in planning_act_seq[(action_num-5):]:
                        cur_edge = train_value["cur"]
                        maxIndex_chosen_index_defined += 1
                    else:
                        cur_edge = train_value["next"]
                        maxIndex_chosen_index_defined = 0
                        planning_act_seq.append(train_value['ans'])
                        action_num += 1
            else:
                cur_edge = train_value["next"]
                maxIndex_chosen_index_defined = 0
                planning_act_seq.append(train_value['ans'])
                action_num += 1
            #if check_stop(train_value['next'], state_test_real_goal):
                #break
        plan_array.append(planning_act_seq)








    saver.save(sess,"./model/GNN-model-"+typea+domain_name+"-n"+data_number+"-p"+save_per+".ckpt")

    index = 0
    for a in plan_array:
        print(str(index)+": \n")
        index = index + 1
        print(a)
