import networkx as nx
import numpy as np


def add_labels(G,meta_path):
    """Summary or Description of the Function

    Parameters:
    argument1 (int): Description of arg1

    Returns:
    int:Returning value

    """
    meta = load_metadata(meta_path) #dictionary
    
    for n in G.nodes():
        G.nodes()[n]["label"] = meta[n]
    
    return(G)



def load_data(path):
    """load the cvs file representing the temporal graph

    Parameters:
    path (string): path of the input dataset

    Returns:
    np.array: a np array with the loaded data

    """
    data = []
    with open(path) as f:
        for line in f:
            tmp = line.split()[0:3]
            arr_tmp = [int(tmp[0]),int(tmp[1]),int(tmp[2])]
        
            data.append(arr_tmp)
    data = np.array(data)
    return(data)



def load_metadata(path_meta):
    """load the metadata of the tmporal graph

    Parameters:
    path_meta (string): path of the meta data

    Returns:
    dict: a dictionary whit key = node and value = attributes

    """
    data = dict()
    with open(path_meta) as f:
        for line in f:
            tmp = line.split()[0:2]

            data[int(tmp[0])] = tmp[1]        

    return(data)



def individuals(data):
    """compute the number of individuals

    Parameters:
    data (np.array): loaded input data

    Returns:
    int: number of individuals in the network

    """
    return(np.unique(data[:,1:]))


def build_graphs(data, gap=0.2, with_labels=False, meta_path=None):
    """build an array of temporal graphs

    Parameters:
    data (np.array): loaded input data
    gap (int): [default = 19] this integer is used to build each snapshot, in particular for each interactions within the temporal graph a new snaphsot is created
    with_labels (boolean): [default = False] if ture, it adds to the temporal snapshots the node label 
    meta_path (str): [default = None] if with_labels is specified, then meta_path is the string of the meta data path

    Returns:
    np.array: an array of static graphs 

    """
    graphs = []
    G = nx.Graph()
    nodes = individuals(data)
    G.add_nodes_from(nodes)
    if with_labels:
        G = add_labels(G, meta_path)

    g = G.copy()

    for time, source, dest in data:
        if dest != 0:
            g.add_edge(source, dest, timestamp=time)
        if nx.density(g) >= gap:
            graphs.append(g)
            g = G.copy()
    if g.number_of_nodes() == 1 or g.number_of_edges() > 0:
        graphs.append(g.copy())

    del g

    return graphs

def build_aggragated_graph(data):
    """compute the agregated network

    Parameters:
    data (np.array): loaded input data

    Returns:
    nx.graph: return a static graph, representing all the interactions

    """
    data = data[:,1:]
    aa = np.unique(data,axis=0)
    G = nx.Graph()
    G.add_edges_from(aa)

    return(G)



def split_input_data(data, gap=19):
    """split the input data according to the temporal gap

    Parameters:
    data (np.array): loaded input data
    gap (int): [default = 19]

    Returns:
    np.array(): return an array of array, the inner array represent the interactions within the gap.

    """
    times = data[:,0]
    pos = times[0]
    chunks = []
    for i in range(len(times)):
        if not(times[i]<=(pos + gap)):
            chunks.append(i)
            pos = times[i]

    return(np.split(data,chunks))

