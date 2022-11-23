import json
import os
import random
from networkx.drawing.nx_pydot import write_dot
from datetime import datetime, timedelta
from info_extraction import extract_posts_users, extract_chains, extract_labels, extract_structure, extract_conversation_graph
from preprocessing import preprocess_trees#, preprocess_text
from stats import create_user_graph, create_discretized_graph
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

#n_graphs = list()
#med_edges = list()
step = 0.1
train_discrete_graphs = dict()
test_discrete_graphs = dict()

for key in train_graph_source.keys():
    train_discrete_graphs[key] = create_discretized_graph(train_graph_source[key], train_graph_dest[key], train_graph_time[key], step)

for key in test_graph_source.keys():
    test_discrete_graphs[key] = create_discretized_graph(test_graph_source[key], test_graph_dest[key], test_graph_time[key], step)

key = random.choice(list(train_discrete_graphs.keys()))
exam = train_discrete_graphs[key]
while len(exam)<6:
    key = random.choice(list(train_discrete_graphs.keys()))
    exam = train_discrete_graphs[key]

pos = nx.circular_layout(exam[0])
i=0

for g in exam:
    #subax1 = plt.subplot(121)
    str_g = "G"+str(i)+".dot"
    write_dot(g, str_g)
    #nx.draw_networkx_nodes(g, node_color='r', label=g.nodes(), pos = pos)
    #nx.draw_networkx_edges(g, edge_color="blue", label="timestamp", pos=nx.circular_layout(g), connectionstyle='arc3, rad = 0.1')

    #plt.show()

'''    
    sum_edges = 0
    for g in list_graph:
        sum_edges = sum_edges+g.number_of_edges()

    n_graphs.append(len(list_graph))
    med_edges.append(sum_edges/len(list_graph))

print(n_graphs)
print(max(n_graphs))
plt.hist(n_graphs, 29)
plt.show()
print(med_edges)
print(max(med_edges))
plt.hist(med_edges, 320,range=[1, 80])
plt.show()

plt.hist(n_nodes)
plt.show()
plt.hist(n_edges)
plt.show()
plt.hist(ratio)
plt.show()
'''