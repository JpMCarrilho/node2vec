import networkx as nx
import tensorflow as tf
import numpy as np

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

    

    return n_G

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
    walk = [u]
    for walk_iter in range(l):
        curr = walk[-1]
        V_curr =  n_G[curr].neighbors()
        s = AliasSample(V_curr,pi)
        walk.append[s]
    return walk


def start_weights(G):
    for edge in G.edges():
        G[edge[0]][edge[1]]['weights'] = 1
    return G

def AliasSample(V_Curr, probs):
    """
    Sampling using alias method
    
    Arguments:
        V_Curr {int} -- current node position
        PreProcessModifedWeights {list} -- list with probabilities of steps being taken
    """
    ##TODO

    return next_state


def alias_setup(probs):
    K       = len(probs)
    q       = np.zeros(K)
    J       = np.zeros(K, dtype=np.int)

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