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
from new_model import HeuristicNetwork
from build_graph import base_graph, make_all_runnable_in_session
import numpy as np
import sonnet as snt
import tensorflow as tf
import datas
from sklearn import svm
from sklearn.svm import NuSVC
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
# domain_name = domain_names[0]
# data_number = "2000"
# percent_number = "0.8"
# dataset = 2000
# save_per = "80"
# test_prob_start = 2001
# test_prob_size = 100
batch_size_tr = 20
dimension = 100
typea="plan"
# Data / training parameters.

num_training_iterations = 20000




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

def getOutputState(output_edges, num_processing_steps, batch_size_tr):
    #output_edges num_step *[g*b]
    res = []
    for num_step in range(num_processing_steps):
        out = np.vsplit(output_edges[num_step],batch_size_tr)
        res_step = []
        for o in out:
            state_proposition = []
            for p in o:
                state_proposition.append(p[0])
            res_step.append(state_proposition)
        res.append(res_step)
    return np.array(res)


def create_heu_loss(output,action):
    # temp = tf.reduce_mean(((output - action)**2))
    temp = tf.subtract(1.0, action)
    tmp = tf.log(tf.clip_by_value(tf.subtract(1.0, output),1e-8,1))

    loss1 = - tf.multiply(action, tf.log(tf.clip_by_value(output,1e-8,1))) - tf.multiply(temp, tmp)
    # loss1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=action))
    loss1sum = tf.reduce_mean(loss1)
    return tf.reduce_sum(loss1sum)





# Data Aquiration
num_processing_steps = 10
action_dict = {}
num_processing_steps, action_dict, grounding = datas.getActionDict(1,dataset+1, domain_name, percent_number,test_prob_start,test_prob_size)
grounding_size = len(grounding)






realstate = tf.placeholder(tf.float32, shape=([num_processing_steps,batch_size_tr*grounding_size,1]))
action_filled = tf.placeholder(tf.float32, shape=([num_processing_steps,batch_size_tr,dimension]))
action_filled_plan = tf.placeholder(tf.float32, shape=([num_processing_steps,1,dimension]))
action_filled_index = tf.placeholder(tf.int32, shape=([num_processing_steps,batch_size_tr]))
action_filled_plan_onestep = tf.placeholder(tf.float32, shape=([1,dimension]))


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

graphs_tuple_goal = utils_tf.placeholders_from_data_dicts(graph_place)






model = new_model.GraphProcess(edge_output_size=1, dimension=dimension)
output_ops_tr, sta_vecs = model(graphs_tuple_ph, action_filled, num_processing_steps)

action_size = len(action_dict)+1


goal_graph = model.getInitialVecForState(graphs_tuple_goal)
training_input, training_output = datas.getTrainData(sta_vecs, goal_graph, action_filled_index, num_processing_steps, batch_size_tr)
# class_weight = None
#clf = RandomForestClassifier(max_depth=200, random_state=0)
clf = svm.SVC(gamma='auto')











# planning
#
#
# get the input and goal graphs
graphs_tuple_test = utils_tf.placeholders_from_data_dicts([base_graph(np.ones(grounding_size), grounding_size, first_action, dimension, nodes)])
goal_graph_tuple_test = utils_tf.placeholders_from_data_dicts([base_graph(np.ones(grounding_size), grounding_size, first_action, dimension, nodes)])
output_ops_tests = model.getInitialVecForState(graphs_tuple_test)
vs0 = output_ops_tests.nodes[0]
goal_ops_test = model.getInitialVecForState(goal_graph_tuple_test)
vsg = goal_ops_test.nodes[0]

# get the most likely action index
concat_array = []
concat_array.append(vs0)
concat_array.append(vs0)
concat_array.append(vsg-vs0)
HeuristicInput = tf.concat(concat_array, axis=-1)
HeuristicInput = tf.stack([HeuristicInput])



progress_onestep_graph, round_progress_onestep_graph = model.processOneStep(output_ops_tests,action_filled_plan_onestep)


init_op = tf.global_variables_initializer()

# saver = tf.train.Saver()

restore_var = tf.trainable_variables()[0:8]
saver = tf.train.Saver(restore_var)

restore_vars = tf.trainable_variables()
savers = tf.train.Saver(restore_vars)



with tf.Session() as sess:
    sess.run(init_op)
    #savers.restore(sess, "./model/GNN-model-"+typea+domain_name+"-n"+data_number+"-p"+save_per+".ckpt")
    saver.restore(sess, "./model/GNN-model-" + domain_name + "-n" + data_number + "-p" + save_per + ".ckpt")
    start_time = time.time()

    # action vector initialization
    action_set_size = len(action_dict)+1

    action_dict["NULL"] = action_set_size-1

    action_storage, proposition_nodes = datas.restoreVectors(domain_name, data_number, save_per)
    pos_dim = (len(action_storage), grounding_size)
    positive_effect = np.ones(pos_dim)
    negative_effect = np.zeros(pos_dim)




    flagstop = 0
    acts_all, state_all, _, _ = datas.getTrainState(1, dataset, num_processing_steps, action_dict, domain_name,
                                            percent_number, grounding)
    start = 0
    stopFlag = False
    training_input_array = []
    training_output_array = []
    for iteration in range(num_training_iterations):
        end = (start + batch_size_tr)%dataset
        st1 = time.time()
        if end < start:
            # acts, state, _, _ = datas.getTrainState(start, dataset+1-start, num_processing_steps, action_dict, domain_name, percent_number,grounding)
            # acts2, state2,_, _ = datas.getTrainState(1, end-1, num_processing_steps, action_dict, domain_name, percent_number, grounding)

            acts = acts_all[start:dataset]
            state = state_all[start:dataset]
            acts2 = acts_all[0:end]
            state2 = state_all[0:end]
            acts = acts + acts2
            state = state + state2
            stopFlag = True
        else:
            # print(end-start)
            # acts, state, _, _ = datas.getTrainState(start, batch_size_tr, num_processing_steps, action_dict, domain_name, percent_number,grounding)
            acts = acts_all[start:end]
            state = state_all[start:end]
        start = end
        st2 = time.time()

        # update action
        act_index = 0
        act_fills = []
        act_fills_index = []
        for num_step in range(num_processing_steps):
            act_fill = []
            act_fill_index = []
            for batch_seq in range(batch_size_tr):
                act_fill.append(action_storage[acts[batch_seq][num_step]])
                act_fill_index.append(acts[batch_seq][num_step])
            act_fills.append(act_fill)
            act_fills_index.append(act_fill_index)
        act_fills = np.array(act_fills)
        act_fills_index = np.array(act_fills_index)
        st3 = time.time()



        state_losses_supervise = getFinalState(state)

        graph_dicts = []
        goal_graph_dicts = []
        for j in range(batch_size_tr):
            input_g = base_graph(state[j][0], grounding_size, first_action, dimension,proposition_nodes)
            input_goal_g = base_graph(state[j][num_processing_steps], grounding_size, first_action, dimension,proposition_nodes)
            graph_dicts.append(input_g)
            goal_graph_dicts.append(input_goal_g)
        st4 = time.time()

        feed_dicts = utils_tf.get_feed_dict(
            graphs_tuple_ph, utils_np.data_dicts_to_graphs_tuple(graph_dicts))
        goal_feed_dicts = utils_tf.get_feed_dict(
            graphs_tuple_goal, utils_np.data_dicts_to_graphs_tuple(goal_graph_dicts))
        feed_dicts.update(goal_feed_dicts)
        feed_dicts[realstate] = state_losses_supervise
        feed_dicts[action_filled] = act_fills
        feed_dicts[action_filled_index] = act_fills_index
        st = time.time()


        train_values = sess.run({
            "outputs": output_ops_tr,
            "training_output":training_output,
            "training_input":training_input
        }, feed_dict=feed_dicts)

        if len(training_input_array) != 0:
            training_input_array = np.concatenate((training_input_array,train_values["training_input"]),axis=0)
            training_output_array = np.concatenate((training_output_array,train_values["training_output"]),axis=0)
        else:
            training_input_array = train_values["training_input"]
            training_output_array = train_values["training_output"]

        if (iteration*batch_size_tr) % dataset > 0:
            round_edges = np.round(train_values['outputs'])
            complete_states = getOutputState(round_edges, num_processing_steps, batch_size_tr)
            for num_step in range(num_processing_steps):
                for batch_index in range(batch_size_tr):
                    if num_step == 0:
                        positive_effect[acts[batch_index][num_step]] = np.multiply(
                            positive_effect[acts[batch_index][num_step]], state[batch_index][0])
                        negative_effect[acts[batch_index][num_step]] = negative_effect[
                                                                           acts[batch_index][num_step]] + np.array(
                            state[batch_index][0])

                    else:
                        positive_effect[acts[batch_index][num_step]] = np.multiply(
                            positive_effect[acts[batch_index][num_step]], complete_states[num_step - 1][batch_index])
                        negative_effect[acts[batch_index][num_step]] = positive_effect[acts[batch_index][num_step]] + \
                                                                       complete_states[num_step - 1][batch_index]



        
        the_time = time.time()
        last_log_time = the_time
        elapsed = time.time() - start_time
        print("# {:05d}, T {:.1f}".format(
            iteration, elapsed))
        if stopFlag:
            break

    clf.fit(training_input_array, training_output_array)

    acts_test, state_test = datas.getTestState(test_prob_start,test_prob_size,num_processing_steps, action_dict, domain_name, percent_number,grounding)
    graph_test_dicts = []
    test_result = []


    plans_test = []
    for iteration in range(test_prob_size):
        graph_test_dicts = []
        goal_graph_test_dicts_array = []
        input_g = base_graph(state_test[iteration][0], grounding_size, first_action, dimension,proposition_nodes)
        graph_test_dicts.append(input_g)
        goal_graph_test_g = base_graph(state_test[iteration][num_processing_steps], grounding_size, first_action, dimension,proposition_nodes)
        goal_graph_test_dicts_array.append(goal_graph_test_g)

        goal_graph_test_dicts = utils_tf.get_feed_dict(
            goal_graph_tuple_test, utils_np.data_dicts_to_graphs_tuple(goal_graph_test_dicts_array))
        np_data_dicts = utils_np.data_dicts_to_graphs_tuple(graph_test_dicts)

        plan_test = []
        num_step = 0
        prev_state_dicts = []
        while num_step < num_processing_steps:
            feed_dicts = utils_tf.get_feed_dict(
                graphs_tuple_test, np_data_dicts)
            feed_dicts.update(goal_graph_test_dicts)



            test_values = sess.run({
                "Input":HeuristicInput,
                "output_edges":output_ops_tests,
                "graph_test":graphs_tuple_test,
                "vs0":vs0,
            }, feed_dict=feed_dicts)
            predict_action = clf.predict(test_values["Input"])
            # get the result of action selection vector

            plan_test.append(predict_action[0])

            act_store = [action_storage[predict_action[0]]]
            feed_dicts[action_filled_plan_onestep]=np.array(act_store)
            progress_result = sess.run({
                "progress_graph":round_progress_onestep_graph,
                "graph_test": graphs_tuple_test,
                "output_edges": output_ops_tests,
            },feed_dict=feed_dicts)
            graph_test_dicts = []
            num_step += 1
            np_data_dicts=(progress_result['progress_graph'])
        plans_test.append(plan_test)
    savers.save(sess,"./model/GNN-model-svm-"+typea+domain_name+"-n"+data_number+"-p"+save_per+".ckpt")

    index = 0
    for a in plans_test:
        print(str(index)+": \n")
        index = index + 1
        print(a)
    









        











    











