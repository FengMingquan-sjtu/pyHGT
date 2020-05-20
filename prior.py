import numpy as np

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def calculate_node_type_prior(node_type_count_dict):
    node_type_list=["paper","author","venue","affiliation","field"]
    prior_list = [1]*len(node_type_list) #smooth
    for k,v in node_type_count_dict.items():
        if k in node_type_list:
            prior_list[node_type_list.index(k)] += v

    prior_array = np.array(prior_list)
    prior_array = softmax(prior_array)

    node_type_prior_dict = {node_type_list[i]:prior_array[i] for i in range(len(prior_array))}
    return node_type_prior_dict



def calculate_relation_type_prior(relation_type_count_dict):
    relation_type_list=["PV_Conference","PV_Journal","PV_Repository","PV_Patent","PP_cite","AP_write_last","AP_write_other","AP_write_first","FF_in,PF_in_L0","PF_in_L3","PF_in_L1","PF_in_L2","PF_in_L5","PF_in_L4","in"]
    prior_list = [1]*len(relation_type_list)
    for k,v in relation_type_count_dict.items():
        if k in relation_type_list:
            prior_list[relation_type_list.index(k)] += v

    prior_array = np.array(prior_list)
    prior_array = softmax(prior_array)

    relation_type_prior_dict = {relation_type_list[i]:prior_array[i] for i in range(len(prior_array))}
    for i in range(len(prior_array)):
        relation_type_prior_dict["rev_"+relation_type_list[i]] = prior_array[i]
    return relation_type_prior_dict

if __name__ == "__main__":
    node_type_count_dict = {"paper":5,"author":6,"venue":2,"affiliation":1,"field":0}
    node_type_prior_dict = calculate_node_type_prior(node_type_count_dict)
    print(node_type_prior_dict)

    relation_type_count_dict = {"PV_Conference":1}
    relation_type_prior_dict =  calculate_relation_type_prior(relation_type_count_dict)
    print(relation_type_prior_dict)

