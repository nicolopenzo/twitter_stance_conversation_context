import networkx as nx

def create_user_graph(list_source, list_dest, list_timestamp):
    G = nx.MultiDiGraph()
    for idx in range(len(list_source)):
        if list_dest[idx] == 0:
            G.add_node(list_source[idx])
        else:
            G.add_edge(list_source[idx], list_dest[idx])
    return G

def create_discretized_graph(list_source, list_dest, list_timestamp, step):
    list_G = list()
    G = nx.MultiDiGraph()
    for idx in range(len(list_source)):
        G.add_node(list_source[idx])

    for idx in range(len(list_source)):
        if list_dest[idx] != 0:
            G.add_edge(list_source[idx], list_dest[idx], timestamp = list_timestamp[idx])
        if nx.density(G) >= step:
            list_G.append(G.copy())
            del G
            G = nx.MultiDiGraph()
            for idx2 in range(len(list_source)):
                G.add_node(list_source[idx2])
    if G.number_of_nodes() == 1 or G.number_of_edges()>0:

        list_G.append(G.copy())
    del G
    return list_G