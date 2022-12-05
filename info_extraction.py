import json
import os
from datetime import timedelta, datetime

def create_tree(struct, id, father):
  tree = list()
  parent = list()

  if struct == []:
    return tree, parent

  if father == 0:
    tree.append(int(id))
    parent.append(int(father))
    tree_child, parent_child = create_tree(struct[id], id, id)
    tree = tree + tree_child
    parent = parent + parent_child
    return tree, parent


  for t in struct.keys():
    tree.append(int(t))
    parent.append(int(father))

  for t in struct.keys():
    tree_child, parent_child = create_tree(struct[t], id, t)
    tree = tree + tree_child
    parent = parent + parent_child

  return tree, parent

def extract_posts_users(path):
    sub_dirs = os.listdir(path)
    posts = dict()
    users = dict()
    os.chdir(path)

    for dir in sub_dirs:
      os.chdir(dir)
      for d in os.listdir("."):
        os.chdir(str(d)+"/source-tweet")
        with open(str(d)+".json", 'r') as f:
          post_info = dict()
          file = json.load(f)
          post_info["text"] = file["text"]
          post_info["user_id"] = file["user"]["id"]
          post_info["in_reply_to_user_id"] = file["in_reply_to_user_id"]
          post_info["created_at"] = datetime.strptime(file["created_at"], '%a %b %d %H:%M:%S %z %Y')

          posts[int(d)] = post_info

          if file["user"]["id"] not in users.keys():
            user_info = dict()
            user_info["description"] = file["user"]["description"]
            user_info["name"] = file["user"]["name"]
            user_info["screen_name"] = file["user"]["screen_name"]
            user_info["created_at"] = datetime.strptime(file["user"]["created_at"], '%a %b %d %H:%M:%S %z %Y')
            users[int(file["user"]["id"])] = user_info
          f.close()

        os.chdir("..")

        os.chdir("replies")
        sons = os.listdir(".")
        if(len(sons)>0):
          for s in sons:
            with open(s, 'r') as f:
              post_info = dict()
              file = json.load(f)
              post_info["text"] = file["text"]

              post_info["in_reply_to_user_id"] = file["in_reply_to_user_id"]
              post_info["created_at"] = datetime.strptime(file["created_at"], '%a %b %d %H:%M:%S %z %Y')
              post_info["user_id"] = file["user"]["id"]
              posts[int(s[:-5])] = post_info

              if file["user"]["id"] not in users.keys():
                user_info = dict()
                user_info["description"] = file["user"]["description"]
                user_info["name"] = file["user"]["name"]
                user_info["screen_name"] = file["user"]["screen_name"]
                user_info["created_at"] = datetime.strptime(file["user"]["created_at"], '%a %b %d %H:%M:%S %z %Y')
                users[int(file["user"]["id"])] = user_info
              f.close()

        os.chdir("..")
        os.chdir("..")



      os.chdir("..")


    os.chdir("..")
    os.chdir("..")
    os.chdir("..")

    return posts, users

def extract_chains(path):
  sub_dirs = os.listdir(path)
  chains = dict()
  os.chdir(path)

  for dir in sub_dirs:
    os.chdir(dir)
    for d in os.listdir("."):
      os.chdir(str(d))
      acc_chain = list()
      acc_chain.append(int(d))
      chains[int(d)] = acc_chain.copy()
      os.chdir("replies")
      sons = os.listdir(".")
      if (len(sons) > 0):
        sons.sort()
        for s in sons:
          acc_chain.append(int(s[:-5]))
          chains[int(s[:-5])] = acc_chain.copy()

      os.chdir("..")
      os.chdir("..")

    os.chdir("..")

  os.chdir("..")
  os.chdir("..")
  os.chdir("..")

  return chains

def extract_labels(path):
  with open(path, 'r') as f:
    data = json.load(f)
    f.close()

  labels = ['comment', 'support', 'deny', 'query']

  keys = dict()

  for d in data['subtaskaenglish']:
    if(len(d)>7):
      keys[int(d)] = labels.index(data['subtaskaenglish'][d])

  return keys

def extract_structure(path):
  sub_dirs = os.listdir(path)

  structures = dict()
  os.chdir(path)
  for dir in sub_dirs:
    os.chdir(dir)
    for d in os.listdir("."):
      os.chdir(str(d))
      with open("structure.json", 'r') as f:
        structures[d] = json.load(f)
        f.close()
      os.chdir("..")

    os.chdir("..")

    # print(structures)

  father = 0
  trees = dict()
  parents = dict()

  for d in structures:
    if len(structures[d]) > 0:
      tree, parent = create_tree(structures[d], d, father)
      for item in tree:
        trees[item] = tree.copy()
        parents[item] = parent.copy()

  os.chdir("..")
  os.chdir("..")
  os.chdir("..")


  return trees, parents


def extract_conversation_graph(trees, posts):
  graph_source = dict()
  graph_post = dict()
  graph_dest = dict()
  graph_time = dict()
  for item in trees.keys():
    sources = list()
    dests = list()
    times = list()
    for idx in range(len(trees[item])):
      sources.append(posts[trees[item][idx]]["user_id"])
      d = posts[trees[item][idx]]["in_reply_to_user_id"]
      if d == None:
        d = 0
      dests.append(d)
      times.append(posts[trees[item][idx]]["created_at"])

    min_time = min(times)
    delta_time = list()
    for t in times:
      delta_time.append(t-min_time)

    graph_source[item] = sources
    graph_dest[item] = dests
    graph_time[item] = delta_time

  return graph_source, graph_dest, graph_time