import datas
import pickle
import sys
domain_names = ["Log","Depot","Zeno","Sat","Ferry","MPrime","grid"]
domain_name = domain_names[int(sys.argv[1])]
data_number = sys.argv[2]
percent_number = sys.argv[3]
dataset = int(data_number)
num_processing_steps = 10
action_dict = {}
num_processing_steps, action_dict, grounding = datas.getActionDict(1,dataset+1, domain_name, percent_number,2001,100)
get_state = []
get_act =[]
for i in range(0, 21):

    _, _ , p_s, p_a = datas.getTrainState(100*i+1, 100, num_processing_steps, action_dict, domain_name, percent_number, grounding)
    get_state += (p_s)
    get_act += p_a
output = open(domain_name+"_"+percent_number+".txt",'wb')
pickle.dump(get_state,output)
pickle.dump(get_act,output)
output.close()

