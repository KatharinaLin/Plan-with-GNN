from prev import ParseFile as PF
import numpy as np
import csv
import tensorflow as tf
def state_data_processing(v3_state, num_processing_steps):
    while len(v3_state) < num_processing_steps+1:
        v3_state=np.row_stack((v3_state,v3_state[len(v3_state)-1]))
    return v3_state

def action_index_sequence_processing(acts, leng, num_processing_steps):
    for act in acts:
        while len(act) < num_processing_steps:
            act.append(leng)

    return acts

def state_complete(state, num_processing_steps):
    extended_state = []
    for i in range(len(state)):
        st = state[i]
        extended_state.append(state_data_processing(st, num_processing_steps))

    return extended_state


def getActionDict(read_from, read_to, domain_name, percent_number,test_prob_start, test_prob_size):
    num_processing_steps = 10
    action_dict = {}
    count = 0
    acts = []
    grounding = []
    for k in range(read_from, read_to):
        filepath = "./prev/solution/"+domain_name+"/"+domain_name+percent_number+"/"+domain_name+"_solution/input_" + str(k) + ".txt"
        print(k)
        _, _, groundings, actions, _, _, _, _ = PF.parsefile(filepath)
        if len(actions) > num_processing_steps:
            num_processing_steps = len(actions)
        if len(groundings) > len(grounding):
            grounding = groundings
        for act in actions:
            if not action_dict.__contains__(act):
                action_dict[act] = count
                count += 1
    for k in range(test_prob_start, test_prob_start+test_prob_size):
        filepath = "./prev/solution/"+domain_name+"/"+domain_name+percent_number+"/"+domain_name+"_solution/input_" + str(k) + ".txt"
        print(k)
        _, _, groundings, actions, _, _, _,_ = PF.parsefile(filepath)
        if len(actions) > num_processing_steps:
            num_processing_steps = len(actions)
        if len(groundings) > len(grounding):
            grounding = groundings
        for act in actions:
            if not action_dict.__contains__(act):
                action_dict[act] = count
                count += 1
    return num_processing_steps, action_dict, grounding

def getTrainState(read_from, read_size, num_processing_steps, action_dict, domain_name, percent_number, grounding):

    acts = []
    state = []
    print_state = []
    print_action = []
    for k in range(read_from, read_from+read_size):
        filepath = "./prev/solution/"+domain_name+"/"+domain_name+percent_number+"/"+domain_name+"_solution/input_" + str(k) + ".txt"
        _, _, groundings, actions, valued3_state, hidden_arr, _, _ = PF.parsefile(filepath)
        state_index = 1
        hidden_valued3_state = []
        hidden_valued3_state.append(valued3_state[0])
        for arr_ele in hidden_arr:
            temp = [0] * len(groundings)
            for ele in arr_ele:
                temp[ele] = 1
            temp1 = np.array(valued3_state[state_index])
            temp = np.array(temp)
            kk = np.multiply(temp1, temp)
            hidden_valued3_state.append(kk)
            state_index += 1
        hidden_valued3_state.append(valued3_state[len(valued3_state)-1])
        print_state.append(hidden_valued3_state)
        print_action.append(actions)
        ex_dim = (len(hidden_valued3_state),len(grounding))
        extended_states = -np.ones(ex_dim)
        if len(grounding) == len(groundings):
            extended_states = hidden_valued3_state
        else:
            for kg in range(len(hidden_valued3_state)):
                for kks in range(len(groundings)):
                    if groundings[kks] in grounding:
                        aka = grounding.index(groundings[kks])
                        extended_states[kg][aka] = hidden_valued3_state[kg][kks]


        action_seq = []
        for action in actions:
            action_seq.append(action_dict[action])
        acts.append(action_seq)
        state.append(extended_states)

    acts = action_index_sequence_processing(acts, len(action_dict), num_processing_steps)
    state = state_complete(state, num_processing_steps)
    return acts, state, print_state, print_action


def getZeroTrainState(read_from, read_size, num_processing_steps, action_dict, domain_name, percent_number, grounding):

    acts = []
    state = []
    print_state = []
    print_action = []
    for k in range(read_from, read_from+read_size):
        filepath = "./prev/solution/"+domain_name+"/"+domain_name+percent_number+"/"+domain_name+"_solution/input_" + str(k) + ".txt"
        _, _, groundings, actions, valued3_state, hidden_arr, _, _= PF.parsefile(filepath)
        state_index = 1
        hidden_valued3_state = []
        hidden_valued3_state.append(valued3_state[0])
        for arr_ele in hidden_arr:
            temp = [0] * len(groundings)
            temp1 = np.array(valued3_state[state_index])
            temp = np.array(temp)
            kk = np.multiply(temp1, temp)
            hidden_valued3_state.append(kk)
            state_index += 1
        hidden_valued3_state.append(valued3_state[len(valued3_state)-1])
        print_state.append(hidden_valued3_state)
        print_action.append(actions)
        ex_dim = (len(hidden_valued3_state),len(grounding))
        extended_states = -np.ones(ex_dim)
        if len(grounding) == len(groundings):
            extended_states = hidden_valued3_state
        else:
            for kg in range(len(hidden_valued3_state)):
                for kks in range(len(groundings)):
                    if groundings[kks] in grounding:
                        aka = grounding.index(groundings[kks])
                        extended_states[kg][aka] = hidden_valued3_state[kg][kks]


        action_seq = []
        for action in actions:
            action_seq.append(action_dict[action])
        acts.append(action_seq)
        state.append(extended_states)

    acts = action_index_sequence_processing(acts, len(action_dict), num_processing_steps)
    state = state_complete(state, num_processing_steps)
    return acts, state, print_state, print_action

def getTestState(test_prob_start, test_prob_size, num_processing_steps, action_dict, domain_name, percent_number, grounding):
    state_test = []
    acts_test = []
    ground = []
    for k in range(test_prob_start, test_prob_start + test_prob_size):
        filepath = "./prev/solution/" + domain_name + "/" + domain_name + percent_number + "/" + domain_name + "_solution/input_" + str(
            k) + ".txt"
        _, _, groundings, actions, valued3_state, _, state_test_real_goal, _ = PF.parsefile(filepath)
        action_seq = []
        for action in actions:
            action_seq.append(action_dict[action])
        acts_test.append(action_seq)
        ex_dim = (len(valued3_state), len(grounding))
        extended_states = np.zeros(ex_dim)
        if len(grounding) == len(groundings):
            extended_states = valued3_state
        else:
            for k in range(len(valued3_state)):
                for kk in range(len(groundings)):
                    if groundings[kk] in grounding:
                        if valued3_state[k][kk] == 1:
                            extended_states[k][grounding.index(groundings[kk])] = 1
        state_test.append(extended_states)
    acts_test = action_index_sequence_processing(acts_test, len(action_dict), num_processing_steps)
    state_test = state_complete(state_test, num_processing_steps)
    return acts_test, state_test

def restoreVectors(domain_name, data_number, save_per):
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
    return restore_action, restore_pn

def getTrainData(train_graphs, goal_graph, action_vector, num_processing_steps, batch_size):
    split_nodes = [tf.split(goal_graph.nodes,goal_graph.n_node, 0)][0]
    goal_graph_state = []
    training_input = []
    training_output = []
    for split_node in split_nodes:
        goal_graph_state.append(split_node[0])
    for num_step in range(num_processing_steps):
        output_graphs = train_graphs[num_step]
        split_nodes = [tf.split(output_graphs.nodes,output_graphs.n_node, 0)][0]
        for batch_index in range(batch_size):
            temp = []
            temp.append(split_nodes[batch_index][0])
            temp.append(split_nodes[batch_index][0])
            temp.append(goal_graph_state[batch_index]-split_nodes[batch_index][0])
            temp = tf.concat(temp, axis=-1)
            training_input.append(tf.stack([temp]))
            training_output.append(tf.stack([action_vector[num_step][batch_index]]))
    training_input = tf.concat(training_input, axis=0)
    training_output = tf.concat(training_output, axis=0)
    return training_input, training_output





