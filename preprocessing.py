import datetime

def preprocess_trees(trees, parents, chains):
    new_trees = dict()
    new_parents = dict()
    for k in chains.keys():
        if k in trees.keys() or k in parents.keys():
            new_tree = list()
            new_parent = list()
            for i in range(len(trees[k])):
                if trees[k][i] in chains[k]:
                    new_tree.append(trees[k][i])
                    new_parent.append(parents[k][i])
            new_trees[k] = new_tree
            new_parents[k] = new_parent

    return new_trees, new_parents
'''
def preprocess_text(posts):
    for post in posts:
        text = post["text"]
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        post["text"] = " ".join(new_text)
    return
'''


def pad_chains(chains, trees, parents, graph_source, graph_dest, graph_time, pad_vector, len_max_chains):
    for idx in range(len(chains)):
        pp = list()
        for j in range(len(chains[idx])):
            pp.append(1)

        pad = len_max_chains-len(chains[idx])

        for p in range(pad):
            chains[idx].append(-1)
            trees[idx].append(-1)
            parents[idx].append(-1)
            graph_source[idx].append(-1)
            graph_dest[idx].append(-1)
            graph_time.append(datetime.timedelta(-1))
            pp.append(0)

        pad_vector.append(pp.copy())
        del pp

    return