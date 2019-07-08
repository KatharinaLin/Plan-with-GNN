import sys
import os
domain_name = "logistics"
dataset =300
from prev import ParseFile as PF
class Action(object):
    def __init__(self, name, para_list, pre_list, eff_list):
        self.name = name
        self.para_list = para_list
        self.pre_list = pre_list
        self.eff_list = eff_list

def delete_brac(line):
    l = line[1:len(line) - 1]
    l = l.strip()
    return l
def parse_action(str_list):
    # parse action name
    index = str_list[0].index(' ') + 1
    name = str_list[0][index:]
    # parse action paralist
    para_list = []
    index = str_list[1].index(' ') + 1
    para_str = str_list[1][index:][1:-1].strip()
    ps = para_str.split('?')[1:]
    for p in ps:
        p_list = []
        p = p.split(' - ')
        para_l = [p[1]]
        if p[1].startswith("(either"):
            paras = delete_brac(p[1].strip())
            para_l = paras.split(' ')[1:]
        if para_l[0][len(para_l[0])-1] == ' ':
            para_l[0] = (para_l[0][0:-1])
        p_list.append(p[0])
        p_list.append((para_l))
        para_list.append((p_list))

    #precondition
    pre_list = []
    index = str_list[2].index(' ') + 1
    pre_str = str_list[2][index:][1:-1].strip()
    if pre_str.startswith('and'):
        pre_str = pre_str.replace("and", '').strip()[1:-1]
    prs = pre_str.split(') (')
    for ps in prs:
        temp = ""
        strs = ps.split(' ?')
        temp = temp+strs[0]
        if len(strs) > 1:
            for i in range(len(strs)-1):
                for j in range(len(para_list)):
                    if para_list[j][0].startswith(strs[i+1]):
                        temp = temp + ' ' + str(j)
                        break

        pre_list.append(temp)
    #eff
    eff_list = []
    index = str_list[3].index(' ') + 1
    eff_str = str_list[3][index:][1:-1].strip()
    if eff_str.startswith('and'):
        eff_str = eff_str.replace("and", '').strip()[1:-1]
    es = eff_str.split(') (')
    for e in es:
        e = e.replace('(', ' ')
        e = e.replace(')', '')
        temp = ""
        strs = e.split(' ?')
        index = 0
        for ss in strs:
            if index == 0:
                index += 1
                continue
            ss = ss.replace(' ','')
            strs[index] = ss
            index += 1
        temp = temp + strs[0]
        if len(strs) > 1:
            for i in range(len(strs) - 1):
                for j in range(len(para_list)):
                    aaa = para_list[j][0]
                    bbb = strs[i + 1]
                    if para_list[j][0].startswith(strs[i + 1]):
                        temp = temp + ' ' + str(j)
                        break

        eff_list.append(temp)
    act = Action(name, para_list, pre_list, eff_list)
    return name, act

def containnumber(str):
    for s in str:
        if s.isdigit():
            return True
    return False
# def progress_one_step(pres, act, Actions, Action_names):

#     act_spl = act.split(' ')
#     act_name = act_spl[0]
#     act_para = act_spl[1:]
#     action = Actions[Action_names.index(act_name)]
#     pre_list = []
#     eff_list = []
#     for pre in action.pre_list:
#         p = ""
#         pre_spl = pre.split(' ')
#         for word in pre_spl:
#             if containnumber(word):
#                 p = p + ' '+ act_para[int(word)]
#             else:
#                 p = p + ' '+word
#         pre_list.append(p.strip())

#     for eff in action.eff_list:
#         e = ""
#         eff_spl = eff.split(' ')
#         for word in eff_spl:
#             if containnumber(word):
#                 e = e + ' '+ act_para[int(word)]
#             else:
#                 e = e + ' '+word
#         eff_list.append(e.strip())
#     res = []
#     not_index_list = []
#     #eff
#     for eff in eff_list:
#         if eff.startswith("not"):
#             eff = eff[4:]
#             if eff in pres:
#                 not_index_list.append(pres.index(eff))
#         else:
#             if eff not in pres:
#                 res.append(eff)
#     for i in range(len(pres)):
#         if i not in not_index_list:
#             res.append(pres[i])
#     return res

def progress_one_step(pres, act, Acts, Action_names, objs, tag=True):
    act_spl = act.split(' ')
    act_name = act_spl[0]
    act_para = act_spl[1:]
    obj_type = []
    for pa in act_para:
        obj = objs[0]
        for obj in objs:
            if obj.startswith(pa):
                break
        obj_type.append(obj[obj.index('-') + 2:])
    for am in Acts:
        if am.name == act_name:
            flag = True
            t_para = am.para_list
            if len(t_para) != len(obj_type):
                continue
            else :
                for o in range(len(obj_type)):

                    if obj_type[o] not in t_para[o][1]:
                        #flag = False
                        return []
            if flag:
                action = am
                break
    #action = Acts[Action_names.index(act_name)]
    t_act_para = action.para_list
    if tag or not tag:

        if len(obj_type) != len(t_act_para):
            return []
        for o in range(len(obj_type)):
            if obj_type[o] not in t_act_para[o][1]:
                return []
    pre_list = []
    eff_list = []
    for pre in action.pre_list:
        p = ""
        pre_spl = pre.split(' ')
        for word in pre_spl:
            if containnumber(word):
                p = p + ' '+ act_para[int(word)]
            else:
                p = p + ' '+word
        pre_list.append(p.strip())
    if tag or not tag:
        for p in pre_list:
            if not p.startswith('not'):
                if p not in pres:
                    return []
            else:
                if p.startswith("not-"):
                    if p not in pres:
                        return []
                p = p[4:]

                if not p.startswith('='):
                    if p in pre:
                        return []
    for eff in action.eff_list:
        e = ""
        eff_spl = eff.split(' ')
        for word in eff_spl:
            if containnumber(word):
                e = e + ' '+ act_para[int(word)]
            else:
                e = e + ' '+word
        eff_list.append(e.strip())
    res = []
    not_index_list = []
    not_pre = []
    add_index = []
    #eff
    for eff in eff_list:
        if eff.startswith("not"):

            eff = eff[4:]

            if eff in pres:
                not_index_list.append(pres.index(eff))
                not_pre.append("not "+eff)
        else:
            if eff not in pres:
                res.append(eff)
                eff_before = "not "+eff
                add_index.append(pres.index(eff_before))
    for i in range(len(pres)):
        if i not in not_index_list:
            if i not in add_index:
                res.append(pres[i])
        else:
            res.append(not_pre[not_index_list.index(i)])

    return res


def progress_plan(init, seq, Actions, Action_names, goal, objs):
    pre_state = init
    states = []
    states.append(init)
    iii = -1
    for a in seq:
        iii += 1
        state = progress_one_step(pre_state, a, Actions, Action_names, objs)
        if len(state) == 0:
            print(iii)
            continue

        states.append(state)
        pre_state = state
        
        
    length_of_goal = len(goal)
    flag = [0]*length_of_goal
    error_num = 0
    Flags = 0
    err_s = []
    index = 0
    for st in states:
        error_num = 0
        for atom in st:
            if atom in goal:
                flag[goal.index(atom)] = 1
        for f in flag:
            if f == 0:
                error_num += 1
        if length_of_goal == 0:
            continue
        error_rate = error_num/length_of_goal
        if error_rate == 0:
            Flags = 1
            print(st)
            break
        index += 1
    return Flags

def readAction(domain_name,percent_number, test_prob_num, test_prob_start, result_file):
    action_dict = {}
    index_dict = {}
    num_processing_steps = 10
    acts=[]
    hidden_bit = []
    count = 0
    for k in range(1, dataset+1):
        filepath = "./prev/solution/"+domain_name+"/"+domain_name+percent_number+"/"+domain_name+"_solution/input_" + str(k) + ".txt"
        _, _, groundings, actions, valued3_state, hidden_arr,_ = PF.parsefile(filepath)

        action_seq = []
        for act in actions:
            if action_dict.__contains__(act):
                action_seq.append(action_dict[act])
            else:
                action_dict[act] = count
                index_dict[count] = act
                count += 1
                action_seq.append(action_dict[act])

        hidden = []
        for arr_ele in hidden_arr:
            temp = [[0]] * len(groundings)
            for ele in arr_ele:
                temp[ele] = [1]
            hidden.append(temp)
        hidden.append([[1]] * len(groundings))
        #for element in result:
            #hidden[element] = [1]
        acts.append(action_seq)
        #print(action_seq)
        hidden_bit.append(hidden)
        grounding = groundings

    # print(len(state))
    # state, state_test, acts, acts_test = train_test_split(state, acts, test_size=0.2)
    acts_test = []
    state_test = []
    print(len(index_dict))
    for k in range(test_prob_start, test_prob_start+test_prob_num):
        print(k)
        filepath = "./prev/solution/"+domain_name+"/"+domain_name+percent_number+"/"+domain_name+"_solution/input_" + str(k) + ".txt"
        _, _, groundings, actions, valued3_state, _ , _= PF.parsefile(filepath)

        state_test.append(valued3_state)
        action_seq = []
        for act in actions:
            if action_dict.__contains__(act):
                action_seq.append(action_dict[act])
            else:
                action_dict[act] = count
                index_dict[count] = act
                count += 1
                action_seq.append(action_dict[act])
        print(action_seq)
        acts_test.append(action_seq)
    result = open(result_file).readlines()
    print(len(index_dict)) 
    len_of_result = len(result)
    l = 0
    flag_number = 0
    action_names = []
    action_array = []
    while l < len_of_result:
        if len(result[l]) < 3:
            l=l+1
            continue
        length = len(result[l])-3
        if result[l][length] == ":":
            action_array.append(action_names)
            action_names = []
            flag_number = int(result[l][:length])
        if result[l][0] == "[":
            allindex = result[l][1:-2]

            allindex = allindex.split(",")

            for index in allindex:
                if int(index) < len(index_dict):
                    action_names.append(index_dict[int(index)])
        l=l+1
    action_array.append(action_names)
    action_array = action_array[1:]


    return action_array
    






def TestModel1(dom_name, test_prob_num, test_prob_start, domain_name,percent_number, action_seqs, key_pre):
    domain = open('./prev/' + dom_name).readlines()
    length_of_domain = len(domain)
    l = 0
    actions = []
    action_names = []
    while l < length_of_domain:
        line = domain[l].strip()

        if line.startswith("(:action"):
            action = []
            action.append(line.lower())
            l = l + 1
            line = domain[l].strip()
            while line != ')':
                if line != "":
                    action.append(line.lower())
                l = l + 1
                line = domain[l].strip()
            action.append(line.lower())
            a_name, act = parse_action(action)
            action_names.append(a_name)
            actions.append(act)
        l = l + 1


    total_error_rate = []
    error = []
    real_goal = []
    for p_n in range(test_prob_num):
        init = []
        goal = []
        action_seq = []
        objs = []
        problem = open('./prev/solution/' + domain_name+"/"+domain_name+percent_number+"/"+domain_name+"_solution/input_"+str(p_n+test_prob_start)+".txt").readlines()
        #problem = open('./'+domain_name+"_solution/input_"+str(p_n+test_prob_start)+".txt").readlines()
        l = 0
        middle_states = []
        while l < len(problem):
            line = problem[l].strip()
            if line.startswith("( :init"):
                l = l+1
                line = problem[l].strip()
                while not line.startswith(')'):
                    init.append(line[0:])
                    l = l + 1
                    line = problem[l].strip()
            if line.startswith("( :objects"):
                l = l + 1
                line = problem[l].strip()
                while not line.startswith(')'):
                    objs.append(line.strip())
                    l = l + 1
                    line = problem[l].strip()
            if line.startswith("( :goal"):
                l = l+1
                line = problem[l].strip()
                while not line.startswith(')'):
                    goal.append(line[0:].strip())
                    l = l + 1
                    line = problem[l].strip()
            if line.startswith("( :states"):
                l = l+1
                line = problem[l].strip()
                temp = []
                while not line.startswith(')'):
                    temp.append(line[0:].strip())
                    l = l + 1
                    line = problem[l].strip()
                middle_states.append(temp)
            if line.startswith(":action"):
                l = l + 1
                action_seq.append(problem[l].strip())
            if line.startswith("(real-goal:"):
                l = l + 1
                line = problem[l].strip()
                while not line.startswith(')'):
                    check = line.strip()
                    if check == '':
                        break
                    real_goal.append(line[0:].strip())
                    l = l+1
                    line = problem[l].strip()
            l = l+1

        #progression
        
        # real_goal = []
        # for g in goal:
        #     if key_pre in g and (not "not" in g):
        #         real_goal.append(g)
        print(p_n,":")
        print(real_goal)
        acc = progress_plan(init, action_seqs[p_n], actions, action_names, real_goal, objs)
        real_goal = []
        total_error_rate.append(acc)
    sumcorrect = 0
    for err in total_error_rate:
        sumcorrect += err
    print("\n")
    return sumcorrect/len(total_error_rate), total_error_rate

def TestModel2(dom_name, test_prob_num, test_prob_start):
    dom_path = "./prev" + dom_name + "/domain.pddl"
    acc_num = 0
    for t in range(test_prob_num):
        problem_path = "./" + dom_name + "_problem/" + str(test_prob_start + t) + ".raw_prob"
        planner = os.popen('./ff -p ./ -o ' + dom_path + ' -f ' + problem_path).read().splitlines()
        for line in planner:
            if "found legal plan as follows" in line:
                acc_num += 1
                break
    return acc_num/test_prob_num


test_prob_start = 2001
test_prob_num = 100
# average_acc_rate2 = TestModel2(domain_name, test_prob_num, test_prob_start)

ind = int(sys.argv[1])
d_name = ["domain_ZenoTravel.pddl","domain_Depot.pddl","domain_logistics.pddl","domain_Sat.pddl","domain_Ferry.pddl","domain_MPrime.pddl","domain_grid.pddl"]
dom_name = d_name[ind]

dom_array = ["Zeno", "Depot","Log","Sat","Ferry","MPrime","grid"]
de_name = dom_array[ind]

domain_name_list = ['Zeno', 'Depot', 'Log','Sat',"Ferry","MPrime","grid"]
domain_name = domain_name_list[ind]

key_pres = ["at", "on", "obj-at","at","at","at","at"]
key_pre = key_pres[ind]
percent_number = sys.argv[2]

result_file = sys.argv[3]
action_seqs=readAction(de_name,percent_number, test_prob_num, test_prob_start, result_file)
result, ins = TestModel1(dom_name, test_prob_num, test_prob_start, domain_name,percent_number, action_seqs, key_pre)
print(result)
print(ins)



