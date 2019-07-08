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


domain_names = ["Log","Depot","Zeno","Sat","Ferry","MPrime","grid"]
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
# data_number = "2000"
# percent_number = "0.6"
# dataset = 2000
# save_per = "60"
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
    errors_precision = []
    errors_recall = []
    errors_acc = []
    f_score = []
    state_tests = state_tests[1:]
    index = 0
    prev = 0
    for output, state in zip(outputs, state_tests):
        if index == 0:
            prev = state
        else:
            if (prev == state).all():
                break
        index += 1
        prev = state
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for out, st in zip(output, state):
            temp = 1
            if st != 1:
                temp = 0
            if temp == 1 and np.round(out[0]) == 1:
                tp += 1
            if temp == 0 and np.round(out[0]) == 0:
                tn += 1
            if temp == 0 and np.round(out[0]) != 0:
                fp += 1
            if temp == 1 and np.round(out[0]) != 1:
                fn += 1
        if fp + tp != 0:
            pre = tp / (fp + tp)
        else:
            pre = 0
        recall = tp / (tp + fn)
        errors_precision.append(pre)
        errors_recall.append(recall)
        errors_acc.append((tp + tn) / (tp + tn + fp + fn))
        if pre + recall != 0:
            f_score.append((2 * pre * recall) / (pre + recall))
        else:
            f_score.append(0)
    return sum(errors_precision) / index, sum(errors_recall) / index, sum(errors_acc) / index, sum(f_score) / index


def create_loss_ops(all_output_ops_tr, length, realstate):

    all_losses = []
    # deal with batch
    for j in range(0,length):
        state = realstate[j]
        output_ops_tr = all_output_ops_tr[j]
        actual_state = tf.nn.relu(state)
        hidden = tf.abs(state)
        temp = tf.subtract(1.0, actual_state)
        tmp = tf.log(tf.clip_by_value(tf.subtract(1.0, output_ops_tr), 1e-8, 1))
        loss1 = - tf.multiply(actual_state, tf.log(tf.clip_by_value(output_ops_tr, 1e-8, 1))) - tf.multiply(temp,tmp)
        loss1 = tf.multiply(loss1, hidden)
        loss1sum = tf.reduce_mean(loss1)
        all_losses.append(loss1sum)
    return tf.reduce_mean(all_losses)
    






# Data Aquiration
num_processing_steps = 1
action_dict = {}
num_processing_steps, action_dict, grounding = datas.getActionDict(1,dataset+1, domain_name, percent_number,test_prob_start,test_prob_size)
grounding_size = len(grounding)






realstate = tf.placeholder(tf.float32, shape=([num_processing_steps,batch_size_tr*grounding_size,1]))
action_filled = tf.placeholder(tf.float32, shape=([num_processing_steps,batch_size_tr,dimension]))
action_filled_test = tf.placeholder(tf.float32, shape=([num_processing_steps,1,dimension]))



# get graphs
first_action = np.random.uniform(-6 / np.sqrt(dimension), 6 / np.sqrt(dimension), dimension)
first_action = np.array(first_action, dtype='float32')

nodes = [1] * ((grounding_size + 1) * dimension)
nodes = np.reshape(nodes, (grounding_size + 1,dimension))
nodes = np.array(nodes, dtype='float32')
graph_place = []
for temp_index in range(batch_size_tr):
    graph_place.append(base_graph(np.ones(grounding_size), grounding_size, first_action, dimension, nodes))
graphs_tuple_ph = utils_tf.placeholders_from_data_dicts(graph_place)

graphs_tuple_test = utils_tf.placeholders_from_data_dicts([base_graph(np.ones(grounding_size), grounding_size, first_action, dimension, nodes)])






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
    savers.restore(sess, "./model/GNN-model-" + domain_name + "-n" + data_number + "-p0" + ".ckpt")
    start_time = time.time()

    # action vector initialization
    action_set_size = len(action_dict)+1

    action_dict["NULL"] = action_set_size-1

   
    action_storage, proposition_nodes = datas.restoreVectors(domain_name, data_number, "0")


    start = 0
    flagstop = 0
    acts_all, state_all, _, _ = datas.getZeroTrainState(1, dataset, num_processing_steps, action_dict, domain_name,
                                                    percent_number, grounding)
    
    
    acts_test, state_test = datas.getTestState(test_prob_start,test_prob_size,num_processing_steps, action_dict, domain_name, percent_number,grounding)
    graph_test_dicts = []
    test_result_precision = []
    test_result_recall = []
    test_acc = []
    test_fscore = []
    for iteration in range(test_prob_size):
        print("it",iteration)
        graph_test_dicts = []
        input_g = base_graph(state_test[iteration][0], grounding_size, first_action, dimension,proposition_nodes)
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
            "output_graphs":sta_vecs_test,
            "aaa":graphs_tuple_test,
        }, feed_dict=feed_dicts)
        pre, rec,acc, fscore = error_calculation(test_values['outputs'], state_test[iteration])
        test_result_precision.append(pre)
        test_result_recall.append(rec)
        test_acc.append(acc)
        test_fscore.append(fscore)
    print("Precision:")
    print(sum(test_result_precision)/(test_prob_size))
    print("Recall:")
    print(sum(test_result_recall)/(test_prob_size))
    print("Accuracy")
    print(sum(test_acc) / (test_prob_size))
    print("F1-score")
    print(sum(test_fscore) / (test_prob_size))
    





        











    











