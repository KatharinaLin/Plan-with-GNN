import datas


domain_names = ["Log","Depot","Zeno"]
dataset = 2100
domain_name = domain_names[1]
test_prob_start = 2001
test_prob_size = 50
percent_number = "0.8"

num_processing_steps, action_dict, grounding = datas.getActionDict(1,dataset+1, domain_name, percent_number,test_prob_start,test_prob_size)