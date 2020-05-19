import sys
from data import *
from utils import *
from model import *
from warnings import filterwarnings
filterwarnings("ignore")
import random
import argparse

def convert(args):
    graph = dill.load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))

    train_range = [t for t in graph.times if t != None and t <= 2015]
    valid_range = [t for t in graph.times if t != None and (t == 2016 or t==2017)]
    test_range  = [t for t in graph.times if t != None and t >= 2018]
    time_range={"fact":train_range, "valid":valid_range, "test":test_range}

    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)    
    pred_file_path = os.path.join(args.out_dir, 'pred.txt')
    pred_file = open(pred_file_path,'w')
    #print(graph.get_types()) #['paper', 'author', 'venue', 'affiliation', 'field']
    #print(graph.get_meta_graph()) #[('venue', 'paper', 'PV_Conference'), ('venue', 'paper', 'PV_Journal'), ('venue', 'paper', 'PV_Repository'), ('venue', 'paper', 'PV_Patent'), ('paper', 'venue', 'rev_PV_Conference'), ('paper', 'venue', 'rev_PV_Journal'), ('paper', 'venue', 'rev_PV_Repository'), ('paper', 'venue', 'rev_PV_Patent'), ('paper', 'paper', 'PP_cite'), ('paper', 'paper', 'rev_PP_cite'), ('paper', 'field', 'rev_PF_in_L0'), ('paper', 'field', 'rev_PF_in_L3'), ('paper', 'field', 'rev_PF_in_L1'), ('paper', 'field', 'rev_PF_in_L2'), ('paper', 'field', 'rev_PF_in_L5'), ('paper', 'field', 'rev_PF_in_L4'), ('paper', 'author', 'AP_write_last'), ('paper', 'author', 'AP_write_other'), ('paper', 'author', 'AP_write_first'), ('field', 'field', 'FF_in'), ('field', 'field', 'rev_FF_in'), ('field', 'paper', 'PF_in_L0'), ('field', 'paper', 'PF_in_L3'), ('field', 'paper', 'PF_in_L1'), ('field', 'paper', 'PF_in_L2'), ('field', 'paper', 'PF_in_L5'), ('field', 'paper', 'PF_in_L4'), ('affiliation', 'author', 'in'), ('author', 'affiliation', 'rev_in'), ('author', 'paper', 'rev_AP_write_last'), ('author', 'paper', 'rev_AP_write_other'), ('author', 'paper', 'rev_AP_write_first')]
    for node_type in graph.get_types():
        pred_file.write("{node_type}(type)\n".format(node_type=node_type))
    for target_type, source_type, relation_type in graph.get_meta_graph():
        if relation_type.startswith("rev_"):
            continue
        pred_file.write("{relation_type}(type,type)\n".format(relation_type=relation_type))
    pred_file.close()

    for split_name in ["fact","valid","test"]:
        time_range_split = time_range[split_name]

        out_file_dir = os.path.join(args.out_dir, '%s_domains' % split_name)
        if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)
        out_file_path = os.path.join(out_file_dir,'1')

        out_file = open(out_file_path,'w')
        node_id_type_set = set()
        for target_type, source_type, relation_type in graph.get_meta_graph():
            if relation_type.startswith("rev_"):
                continue
            tmp_dict = graph.edge_list[target_type][source_type][relation_type]
            target_ids = tmp_dict.keys()
            for target_id in target_ids:
                if random.random() > 0.02:
                    continue
                node_id_type_set.add((target_id,target_type))
                source_ids = tmp_dict[target_id].keys()
                for source_id in source_ids:
                    time = tmp_dict[target_id][source_id]
                    if time in time_range_split: # if this time falls in this time range split, then record
                        node_id_type_set.add((source_id,source_type))
                        out_file.write("1\t{relation_type}({source_id},{target_id})\n".format(relation_type=relation_type,source_id=source_id,target_id=target_id))


        for node_id, node_type in node_id_type_set:
            if random.random() > 0.05:
                continue
            out_file.write("1\t{node_type}({node_id})\n".format(node_type=node_type,node_id=node_id))

        out_file.close()

                


    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert OAG into factor graph')
    parser.add_argument('--data_dir', type=str, default='./dataset/oag_output',
                    help='The address of preprocessed graph.')
    parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medicion or All: _CS or _Med or (empty)') 
    parser.add_argument('--out_dir', type=str, default='./data_save',
                    help='The address for storing factor graph.')    
    args = parser.parse_args()
    convert(args)
