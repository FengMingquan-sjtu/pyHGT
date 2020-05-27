import sys
from data import *
from utils import *
from model import *
from warnings import filterwarnings
filterwarnings("ignore")
import random
import argparse

def inspect(args):
    graph = dill.load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))
    node_types = ['paper', 'author', 'venue', 'affiliation', 'field']
    edge_types = ["PV","PP","PF","AP","FF"]
    num_node = {n_type : len(graph.node_feature[n_type]) for n_type in node_types}
    print(num_node)
    print(sum([v for k,v in num_node.items()]))

    num_edge_types = {e_type:0 for e_type in edge_types}
    for target_type, source_type, relation_type in graph.get_meta_graph():
        for e_type in edge_types:
            if relation_type.startswith(e_type):
                tmp_dict = graph.edge_list[target_type][source_type][relation_type]
                tmp_sum = 0
                for target_id, source_ids in tmp_dict.items():
                    tmp_sum += len(source_ids)
                num_edge_types[e_type] += tmp_sum 
                break
    print(num_edge_types)
    print(sum([v for k,v in num_edge_types.items()]))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inspect OAG dataset')
    parser.add_argument('--data_dir', type=str, default='./dataset/oag_output',
                    help='The address of preprocessed graph.')
    parser.add_argument('--domain', type=str, default='_CS',
                    help='CS, Medicion or All: _CS or _Med or (empty)') 
  
    args = parser.parse_args()
    inspect(args)
