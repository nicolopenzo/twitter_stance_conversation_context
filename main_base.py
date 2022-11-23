import json
import os
from datetime import datetime, timedelta
from info_extraction import extract_posts_users, extract_chains, extract_labels, extract_structure, extract_conversation_graph
from preprocessing import preprocess_trees#, preprocess_text
from stats import create_user_graph
import networkx as nx
import matplotlib.pyplot as plt

train_path = "rumoureval2019/rumoureval-2019-training-data/twitter-english/"

test_path = "rumoureval2019/rumoureval-2019-test-data/twitter-en-test-data/"

train_labels = extract_labels('rumoureval2019/rumoureval-2019-training-data/train-key.json')
dev_labels = extract_labels('rumoureval2019/rumoureval-2019-training-data/dev-key.json')
test_labels = extract_labels('rumoureval2019/final-eval-key.json')

train_posts, train_users = extract_posts_users(train_path)
test_posts, test_users = extract_posts_users(test_path)

train_chains = extract_chains(train_path)
test_chains = extract_chains(test_path)

train_trees, train_parents = extract_structure(train_path)
test_trees, test_parents = extract_structure(test_path)

train_fin_trees, train_fin_parents = preprocess_trees(train_trees, train_parents, train_chains)
test_fin_trees, test_fin_parents = preprocess_trees(test_trees, test_parents, test_chains)

del train_trees
del train_parents
del test_trees
del test_parents

#preprocess_text(train_posts)
#preprocess_text(test_posts)

train_graph_source, train_graph_dest, train_graph_time = extract_conversation_graph(train_fin_trees, train_posts)
test_graph_source, test_graph_dest, test_graph_time = extract_conversation_graph(test_fin_trees, test_posts)

max_nodes = 0
max_graph = None
max_graph_ratio = None
max_id = 0
max_ratio_id = 0
n_nodes = list()
n_edges = list()
ratio = list()
max_ratio = 0
for key in train_graph_source.keys():
    graph = create_user_graph(train_graph_source[key], train_graph_dest[key], train_graph_time[key])
    n_nodes.append(graph.number_of_nodes())
    n_edges.append(graph.number_of_edges())
    ratio.append(graph.number_of_edges()/graph.number_of_nodes())
    if(max_nodes < graph.number_of_nodes()):
        max_nodes = graph.number_of_nodes()
        max_graph = graph.copy()
        max_id = key
    if (max_ratio < (graph.number_of_edges()/graph.number_of_nodes())):
        print("TEST")
        max_ratio = graph.number_of_edges()/graph.number_of_nodes()
        print(max_ratio)
        print(key)
        max_ratio_id = key
        max_graph_ratio = graph.copy()
    del graph

print("MAXNODES")
print(max_nodes)
print(max_id)
print("MAXEDGES")
print(max(n_edges))
print("MAXRATIO")
print(max(ratio))
print(max_ratio_id)
print(n_nodes[ratio.index(max(ratio))])
print(n_edges[ratio.index(max(ratio))])
subax1 = plt.subplot(121)

nx.draw(max_graph_ratio, node_color='r', edge_color='b')

plt.show()

'''
plt.hist(n_nodes)
plt.show()
plt.hist(n_edges)
plt.show()
plt.hist(ratio)
plt.show()
'''