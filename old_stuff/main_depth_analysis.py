import json
import os

def create_tree(struct, id, father, depth):
  tree = list()
  parent = list()

  if struct == []:
    return tree, parent

  if father == 0:
    tree.append(id)
    parent.append(depth+1)
    tree_child, parent_child = create_tree(struct[id], id, id, depth+1)
    tree = tree + tree_child
    parent = parent + parent_child
    return tree, parent


  for t in struct.keys():
    tree.append(t)
    parent.append(depth+1)

  for t in struct.keys():
    tree_child, parent_child = create_tree(struct[t], id, t, depth+1)
    tree = tree + tree_child
    parent = parent + parent_child

  return tree, parent

workdir = os.getcwd()
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



for d in structures:
  tree, parent = create_tree(structures[d], d, father, 0)
  trees.append(tree)
  parents.append(max(parent))

print("MAXDEPTH")
print(max(parents))


os.chdir(workdir)

with open('rumoureval2019/final-eval-key.json', 'r') as f:
  data_test = json.load(f)

count_test = [0, 0, 0, 0]

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

for d in structures:
  tree, parent = create_tree(structures[d], d, father, 0)
  trees_test.append(tree)
  parents_test.append(max(parent))
  print(parents_test)

print("MAXDEPTH")
print(max(parents_test))
print(len(parents_test))


os.chdir(workdir)