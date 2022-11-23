import json
import os

def create_tree(struct, id, father):
  tree = list()
  parent = list()

  if struct == []:
    return tree, parent

  if father == 0:
    tree.append(id)
    parent.append(father)
    tree_child, parent_child = create_tree(struct[id], id, id)
    tree = tree + tree_child
    parent = parent + parent_child
    return tree, parent


  for t in struct.keys():
    tree.append(t)
    parent.append(father)

  for t in struct.keys():
    tree_child, parent_child = create_tree(struct[t], id, t)
    tree = tree + tree_child
    parent = parent + parent_child

  return tree, parent

with open('rumoureval2019/rumoureval-2019-training-data/train-key.json', 'r') as f:
  data = json.load(f)

labels = list()

workdir = os.getcwd()

'''
for d in data['subtaskaenglish']:
  #print(d, data['subtaskaenglish'][d])
  if data['subtaskaenglish'][d] not in labels:
    labels.append( data['subtaskaenglish'][d])

print(labels)
['comment', 'support', 'deny', 'query'] --> [0, 1, 2, 3]
'''

labels = ['comment', 'support', 'deny', 'query']
count_train = [0, 0, 0, 0]

train_keys = dict()

for d in data['subtaskaenglish']:
  train_keys[d] = labels.index(data['subtaskaenglish'][d])
  count_train[train_keys[d]] = count_train[train_keys[d]]+1

#print(train_keys)
print(count_train)

with open('rumoureval2019/rumoureval-2019-training-data/dev-key.json', 'r') as f:
  data_dev = json.load(f)

count_dev = [0, 0, 0, 0]

dev_keys = dict()

for d in data_dev['subtaskaenglish']:
  dev_keys[d] = labels.index(data_dev['subtaskaenglish'][d])
  count_dev[dev_keys[d]] = count_dev[dev_keys[d]]+1

#print(test_keys)
print(count_dev)

sub_dirs = os.listdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")

structures = dict()
os.chdir("rumoureval2019/rumoureval-2019-training-data/twitter-english/")
for dir in sub_dirs:
  os.chdir(dir)
  for d in os.listdir(".."):
    os.chdir(str(d))
    with open("structure.json", 'r') as f:
      structures[d] = json.load(f)
    os.chdir("../..")

  os.chdir("../..")

  #print(structures)

father = 0
trees = list()
parents = list()
count = [0,0,0,0]

t_keys = train_keys | dev_keys

for d in structures:
  #print(d)
  count[t_keys[d]] = count[t_keys[d]] + 1
  tree, parent = create_tree(structures[d], d, father)
  trees.append(tree)
  parents.append(parent)
'''
print("TREE")
print(trees)

print("PARENTS")
print(parents)
'''
print(count)#[11, 303, 11, 0]

os.chdir(workdir)

with open('rumoureval2019/final-eval-key.json', 'r') as f:
  data_test = json.load(f)

count_test = [0, 0, 0, 0]

test_keys = dict()

for d in data_test['subtaskaenglish']:
  test_keys[d] = labels.index(data_test['subtaskaenglish'][d])
  test_keys[d] = labels.index(data_test['subtaskaenglish'][d])
  count_test[test_keys[d]] = count_test[test_keys[d]]+1

print(count_test)#[1476, 157, 101, 93]

sub_dirs = os.listdir("rumoureval2019/rumoureval-2019-test-data/twitter-en-test-data/")

structures = dict()
os.chdir("rumoureval2019/rumoureval-2019-test-data/twitter-en-test-data/")
for dir in sub_dirs:
  os.chdir(dir)
  for d in os.listdir(".."):
    os.chdir(str(d))
    with open("structure.json", 'r') as f:
      structures[d] = json.load(f)
    os.chdir("../..")

  os.chdir("../..")

  #print(structures)

father = 0
trees_test = list()
parents_test = list()
count_test = [0,0,0,0]

for d in structures:
  #print(d)
  count_test[test_keys[d]] = count_test[test_keys[d]] + 1
  tree, parent = create_tree(structures[d], d, father)
  trees_test.append(tree)
  parents_test.append(parent)
'''
print("TREE")
print(trees)

print("PARENTS")
print(parents)
'''
print(count_test)#[0, 50, 0, 6]

os.chdir(workdir)