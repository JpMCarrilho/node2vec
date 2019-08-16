import networkx as nx
import tensorflow as tf
import numpy as np
import numpy.random as npr

def PreProcessModifedWeights(G,p,q):
    """
    Process weights for Alias sampling
    
    Arguments:
        G {Graph} -- Graph to be processed
        p {float} -- Return parameter of walk
        q {float} -- In-Out parameter of walk
    
    Returns:
        n_G -- updated graph
    """
    alias_nodes =[]
    for node in G.nodes():
        probs = [G[node][nbr]['weight'] for nbr in G.neighbors()]
        Z = sum(probs) ##normalizing constant
        normalized_probs = [prob/Z for prob in probs]
        alias_nodes[node] = alias_setup(normalized_probs)
    
    alias_edges = []
    
    for edge in G.edges():
        alias_edges[edge] = get_alias_edge(edge[0],edge[1],p,q)
        alias_edges[edge[1],edge[0]] = get_alias_edge[edge[1],edge[0],p,q]


    

    return alias_nodes, alias_edges

def node2VecWalk(n_G,u,l):
    """
    Simulates Biased Random Walk from node2vec paper

    
    Arguments:
        n_G {Graph} -- graph to be used in simulation
        u {int} -- starting node for r-walk
        l {int} -- length of the walk
    Returns:
        walk -- generated biased random walk
    """
    for node in G.nodes()
    
    walk = [u]
    for walk_iter in range(l):
        curr = walk[-1]
        V_curr =  n_G[curr].neighbors()
        s = alias_draw() ##AliasSample()
        walk.append[s]
    return walk


def start_weights(G):
    for edge in G.edges():
        G[edge[0]][edge[1]]['weights'] = 1
    return G


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    K  = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand()*K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]

# Construct the table.
#J, q = alias_setup(probs)

def get_alias_edge(G,src,dst,p,q):
    """
    processes edges for alias sampling
    
    Arguments:
        G {graph} -- graph being processed
        src {int} -- source node from the edge
        dst {int} -- destination node from the edge
        p {float} -- Return parameter of the biased random walk
        q {float} -- In-Out parameter of the biased random walk
    
    Returns:
        alias_setup(normalized_probs) -- normalized probabilitry distribution of edges for biased random walk
    """
    probs = []
    for nbr in dst.neighbors():
        if nbr == src:
            prob = G[nbr][dst]['weight']/p
        elif G.has_edge(src,nbr):
            prob = G[src][nbr] = 1
        else:
            prob = G[nbr][dst]['weight']/q
        probs.append(prob)
    Z = sum(probs) ##normalizing constant
    normalized_probs = [prob/Z for prob in probs]

    return alias_setup(normalized_probs)

