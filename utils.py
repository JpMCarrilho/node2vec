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
        s = AliasSample(V_curr,PreProcessModifedWeights(n_G,p,q))
        walk.append[s]
    return walk

def start_weights(G):
    for edge in G.edges():
        G[edge[0]][edge[1]]['weights'] = 1
    return G
    