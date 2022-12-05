import json
import os
import random
from networkx.drawing.nx_pydot import write_dot
from datetime import datetime, timedelta
from sklearn.metrics import classification_report
from collections import defaultdict
import construction as cs
from ETN import *
from ETMM import *
from info_extraction import extract_posts_users, extract_chains, extract_labels, extract_structure, extract_conversation_graph
from preprocessing import preprocess_trees#, preprocess_text
import numpy as np
from stats import create_user_graph, create_discretized_graph
from transformers import BertweetTokenizer
from dataset_conversations import TweetConversations, create_data_loader
import torch
from model_conversations import TweetConversationModel
from model_loop_BERT import train_step, eval_model, get_predictions
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
'''
for key in train_graph_source.keys():
    train_discrete_graphs[key] = create_discretized_graph(train_graph_source[key], train_graph_dest[key], train_graph_time[key], step)

for key in test_graph_source.keys():
    test_discrete_graphs[key] = create_discretized_graph(test_graph_source[key], test_graph_dest[key], test_graph_time[key], step)
'''
del_keys = list()
for key in dev_labels.keys():
    if key not in train_fin_trees.keys():
        del_keys.append(key)

for k in del_keys:
    dev_labels.pop(k)

data = dict()


for key in train_graph_source.keys():
    data_raw = np.zeros((len(train_graph_source[key]), 4), dtype=np.int)
    for idx in range(len(train_graph_source[key])):
        data_raw[idx, 0] = int(train_graph_time[key][idx])
        data_raw[idx, 1] = int(train_graph_source[key][idx])
        data_raw[idx, 2] = int(train_graph_dest[key][idx])
        post = train_fin_trees[key][idx]
        if post in train_labels.keys():
            label = train_labels[post]
        else:
            label = dev_labels[post]

        data_raw[idx, 3] = int(label)

    data[key] = data_raw
    string_line = ""
    for idx in range(data_raw.shape[0]):
        string_line = string_line + str(data_raw[idx, 0]) + " " + str(data_raw[idx, 1]) + " " + str(data_raw[idx, 2]) + " " + str(data_raw[idx, 3]) + "\n"
    string_line = string_line[:-1]
    text_file = open("graphs/" + str(key) + ".txt", "wt")
    n = text_file.write(string_line)
    text_file.close()


'''
firms = dict()
for key in data.keys():
'''


'''
tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base")
MAX_LEN_BERT = 130
BATCH_SIZE = 15



print("Creating Training set...")
train_loader = create_data_loader(train_posts, train_users, train_chains, train_fin_trees, train_fin_parents,
                                  train_graph_source, train_graph_dest, train_graph_time, train_discrete_graphs,
                                  tokenizer, MAX_LEN_BERT,  train_labels, BATCH_SIZE)

dev_loader = create_data_loader(train_posts, train_users, train_chains, train_fin_trees, train_fin_parents,
                                  train_graph_source, train_graph_dest, train_graph_time, train_discrete_graphs,
                                  tokenizer, MAX_LEN_BERT,  dev_labels, BATCH_SIZE)


test_loader = create_data_loader(test_posts, test_users, test_chains, test_fin_trees, test_fin_parents,
                                  test_graph_source, test_graph_dest, test_graph_time, test_discrete_graphs,
                                  tokenizer, MAX_LEN_BERT,  test_labels, BATCH_SIZE)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# hyper parameters
size = 50
dropout = 0.5
lr = 0.05
weight_decay = 1e-4
epochs = 30

# create the model
model = TweetConversationModel(4, device).to(device)
# create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=weight_decay)

# training loop
total_steps = len(train_loader)*epochs
loss_fn = torch.nn.CrossEntropyLoss().to(device)

history = defaultdict(list)
best_loss = 1000000.
count = 0

for epoch in range(epochs):
    model = model.train()
    losses = []
    correct_predictions = 0.0
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)
    for data in train_loader:
        train_acc, train_loss = train_step(model, data, loss_fn, optimizer, device, len(data['labels']))
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, dev_loader, loss_fn, device, len(dev_labels))
        print(f'Val loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_loss < best_loss:
            torch.save(model, 'model.pth')
            best_loss = val_loss
            count = 0
        else:
            count += 1
            if count == 60 and epoch > 0:
                break
    if count == 60 and epoch > 0:
        break

model = torch.load('model.pth')

test_acc, test_loss = eval_model(model, test_loader, loss_fn, device, len(test_labels))
print(f'Test loss {test_loss} accuracy {test_acc}')

y_text, y_pred, y_pred_probs, y_test = get_predictions(model, test_loader, device)

print(classification_report(y_test, y_pred, digits=4))
'''