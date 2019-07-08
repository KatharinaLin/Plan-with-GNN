from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import sys
import numpy as np

import utils_tf
import utils_np
import new_model

from prev import ParseFile as PF
from models import HeuristicNetwork
from build_graph import base_graph, make_all_runnable_in_session
import numpy as np
import sonnet as snt
import tensorflow as tf
import datas

import os
import sys
import csv

domain_names = ["Log", "Depot", "Zeno", "Sat", "Ferry", "MPrime", "grid"]
domain_name = domain_names[int(sys.argv[1])]
data_number = sys.argv[2]

percent_number = sys.argv[3]
dataset = int(data_number)
save_per = sys.argv[4]
test_prob_start = int(sys.argv[5])
test_prob_size = int(sys.argv[6])
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[7]
# domain_name = domain_names[4]
#
# data_number = "20"
# percent_number = "0.8"
# dataset = 20
# save_per = "80"
# test_prob_start = 2001
# test_prob_size = 100
batch_size_tr = 20
dimension = 100
# Data / training parameters.
num_training_iterations = 30000

try:
    import seaborn as sns
except ImportError:
    pass
else:
    sns.reset_orig()

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)


def getFinalState(state_batch):
    finals = []
    num_steps = len(state_batch[0])
    batch_size = len(state_batch)
    for j in range(1, num_steps):
        final_array = []
        for k in range(batch_size):

            for element in state_batch[k][j]:
                final_array.append([element])
        finals.append(final_array)
    return finals


def error_calculation(outputs, state_tests):
    errors = []
    state_tests = state_tests[1:]
    for output, state in zip(outputs, state_tests):
        error_count = []
        for out, st in zip(output, state):
            temp = 1
            if st != 1:
                temp = 0
            error_count.append(np.abs(np.round(out[0]) - temp))
        errors.append(sum(error_count) / len(state))
    return sum(errors) / len(state_tests)


def create_loss_ops(all_output_ops_tr, length, realstate):
    all_losses = []
    # deal with batch
    for j in range(0, length):
        state = realstate[j]
        output_ops_tr = all_output_ops_tr[j]
        actual_state = tf.nn.relu(state)
        hidden = tf.abs(state)
        temp = tf.subtract(1.0, actual_state)
        tmp = tf.log(tf.clip_by_value(tf.subtract(1.0, output_ops_tr), 1e-8, 1))
        loss1 = - tf.multiply(actual_state, tf.log(tf.clip_by_value(output_ops_tr, 1e-8, 1))) - tf.multiply(temp, tmp)
        loss1 = tf.multiply(loss1, hidden)
        loss1sum = tf.reduce_mean(loss1)
        all_losses.append(loss1sum)
    return tf.reduce_mean(all_losses)


# Data Aquiration
num_processing_steps = 1
action_dict = {}
num_processing_steps, action_dict, grounding = datas.getActionDict(1, dataset + 1, domain_name, percent_number,
                                                                   test_prob_start, test_prob_size)
grounding_size = len(grounding)

realstate = tf.placeholder(tf.float32, shape=([num_processing_steps, batch_size_tr * grounding_size, 1]))
action_filled = tf.placeholder(tf.float32, shape=([num_processing_steps, batch_size_tr, dimension]))
action_filled_test = tf.placeholder(tf.float32, shape=([num_processing_steps, 1, dimension]))

# get graphs
first_action = np.random.uniform(-6 / np.sqrt(dimension), 6 / np.sqrt(dimension), dimension)
first_action = np.array(first_action, dtype='float32')

nodes = [1] * ((grounding_size + 1) * dimension)
nodes = np.reshape(nodes, (grounding_size + 1, dimension))
nodes = np.array(nodes, dtype='float32')
graph_place = []
for temp_index in range(batch_size_tr):
    graph_place.append(base_graph(np.ones(grounding_size), grounding_size, first_action, dimension, nodes))
graphs_tuple_ph = utils_tf.placeholders_from_data_dicts(graph_place)

graphs_tuple_test = utils_tf.placeholders_from_data_dicts(
    [base_graph(np.ones(grounding_size), grounding_size, first_action, dimension, nodes)])

model = new_model.GraphProcess(edge_output_size=1, dimension=dimension)
output_ops_tr, sta_vecs = model(graphs_tuple_ph, action_filled, num_processing_steps)

output_ops_test, sta_vecs_test = model(graphs_tuple_test, action_filled_test, num_processing_steps)

# Training Loss

loss_ops_trs = []
loss_ops_tr = create_loss_ops(output_ops_tr, num_processing_steps, realstate)

loss_op_tr = tf.reduce_sum(loss_ops_tr)
# Optimizer.
params = tf.trainable_variables()
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
restore_vars = tf.trainable_variables()
savers = tf.train.Saver(restore_vars)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # savers.restore(sess, "./model/GNN-model-" + domain_name + "-n" + data_number + "-p" + save_per + ".ckpt")
    start_time = time.time()

    # action vector initialization
    action_set_size = len(action_dict) + 1

    action_dict["NULL"] = action_set_size - 1

    action_storage = []
    for i in range(action_set_size + 1):
        temp_action = np.random.uniform(-6 / np.sqrt(dimension), 6 / np.sqrt(dimension), dimension)
        action_storage.append(temp_action)

    proposition_nodes = (
        np.random.uniform(-6 / np.sqrt(dimension), 6 / np.sqrt(dimension), (grounding_size + 1) * dimension))
    proposition_nodes = np.reshape(proposition_nodes, (grounding_size + 1, dimension))
    proposition_nodes = np.array(proposition_nodes, dtype='float32')
    # action_storage, proposition_nodes = datas.restoreVectors(domain_name, data_number, save_per)

    start = 0
    flagstop = 0
    acts_all, state_all, _, _ = datas.getZeroTrainState(1, dataset, num_processing_steps, action_dict, domain_name,
                                                    percent_number, grounding)
    for iteration in range(num_training_iterations):
        end = (start + batch_size_tr) % dataset
        if end <= start:
            acts = acts_all[start:dataset]
            state = state_all[start:dataset]
            acts2 = acts_all[0:end]
            state2 = state_all[0:end]
            acts = acts + acts2
            state = state + state2

        else:
            # print(end-start)
            # acts, state, _, _ = datas.getTrainState(start, batch_size_tr, num_processing_steps, action_dict, domain_name, percent_number,grounding)
            acts = acts_all[start:end]
            state = state_all[start:end]
        start = end

        # update action
        act_index = 0
        act_fills = []
        for num_step in range(num_processing_steps):
            act_fill = []
            for batch_seq in range(batch_size_tr):
                act_fill.append(action_storage[acts[batch_seq][num_step]])
            act_fills.append(act_fill)
        act_fills = np.array(act_fills)

        state_losses_supervise = getFinalState(state)
        graph_dicts = []
        for j in range(batch_size_tr):
            input_g = base_graph(state[j][0], grounding_size, first_action, dimension, proposition_nodes)
            graph_dicts.append(input_g)
        feed_dicts = utils_tf.get_feed_dict(
            graphs_tuple_ph, utils_np.data_dicts_to_graphs_tuple(graph_dicts))
        feed_dicts[realstate] = state_losses_supervise
        feed_dicts[action_filled] = act_fills

        train_values = sess.run({
            "step": step_op,
            "loss": loss_op_tr,
            "outputs": output_ops_tr,
            "output_graphs": sta_vecs,
            "test": action_filled,
        }, feed_dict=feed_dicts)
        # update action storage
        act_index = 0
        act_fills = []

        for num_step in range(num_processing_steps):
            output_graphs_actions = train_values['output_graphs'][num_step + 1].globals
            for batch_seq in range(batch_size_tr):
                action_storage[acts[batch_seq][num_step]] = output_graphs_actions[batch_seq]

        proposition_nodes = train_values['output_graphs'][0].nodes[:grounding_size + 1]
        the_time = time.time()
        last_log_time = the_time
        elapsed = time.time() - start_time
        print("# {:05d}, T {:.1f}, Ltr {:6f}".format(
            iteration, elapsed, train_values["loss"]))
        if iteration % 5000 == 0:
            saver.save(sess, "./model/GNN-model-" + domain_name + "-n" + data_number + "-p0" + ".ckpt")
        if train_values["loss"] < 5 * 1e-3:
            flagstop += 1
            if flagstop > (dataset / batch_size_tr):
                break
        else:
            if flagstop > 0:
                flagstop = 0

    saver.save(sess, "./model/GNN-model-" + domain_name + "-n" + data_number + "-p0" + ".ckpt")
    with open(domain_name + "-n" + data_number + "-p0" + "_action.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for arr_ele in action_storage:
            writer.writerow(arr_ele)
    with open(domain_name + "-n" + data_number + "-p0" + "_pnode.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for arr_ele in proposition_nodes:
            writer.writerow(arr_ele)
    acts_test, state_test = datas.getTestState(test_prob_start, test_prob_size, num_processing_steps, action_dict,
                                               domain_name, percent_number, grounding)
    graph_test_dicts = []
    test_result = []
    for iteration in range(test_prob_size):
        graph_test_dicts = []
        input_g = base_graph(state_test[iteration][0], grounding_size, first_action, dimension, proposition_nodes)
        graph_test_dicts.append(input_g)
        feed_dicts = utils_tf.get_feed_dict(
            graphs_tuple_test, utils_np.data_dicts_to_graphs_tuple(graph_test_dicts))
        # update action
        act_index = 0
        act_fills = []

        for num_step in range(num_processing_steps):
            act_fills.append([action_storage[acts_test[iteration][num_step]]])
        act_fills = np.array(act_fills)
        feed_dicts[action_filled_test] = act_fills
        test_values = sess.run({
            "outputs": output_ops_test,
            "output_graphs": sta_vecs_test,
            "aaa": graphs_tuple_test,
        }, feed_dict=feed_dicts)
        test_result.append(error_calculation(test_values['outputs'], state_test[iteration]))
    print("Test:")
    print(sum(test_result) / (test_prob_size))





























