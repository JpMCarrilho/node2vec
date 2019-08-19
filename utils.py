import networkx as nx
import tensorflow as tf
import numpy as np
import numpy.random as npr

def PreProcessModifedweight(G,p,q):
    """
    Process weight for Alias sampling
    
    Arguments:
        G {Graph} -- Graph to be processed
        p {float} -- Return parameter of walk
        q {float} -- In-Out parameter of walk
    
    Returns:
        n_G -- updated graph
    """
    alias_nodes = {}
    G = G.to_undirected()
    print(G.edges(data=True))
    for node in G.nodes():
        
        probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
        print(len(sorted(G.neighbors(node))))
        #print(G.neighbors(node))
        Z = sum(probs) ##normalizing constant
        #print(probs)
        normalized_probs = [float(prob)/Z for prob in probs]
        #print(normalized_probs)
        alias_nodes[node] = alias_setup(normalized_probs)
    
        
    
    alias_edges = {}
    
    for edge in G.edges():
        alias_edges[edge] = get_alias_edge(G,edge[0],edge[1],p,q)
        alias_edges[edge[1],edge[0]] = get_alias_edge(G,edge[1],edge[0],p,q)

    return alias_nodes, alias_edges

def node2VecWalk(n_G,u,l,alias_nodes,alias_edges):
    """
    Simulates Biased Random Walk from node2vec paper

    
    Arguments:
        n_G {Graph} -- graph to be used in simulation
        u {int} -- starting node for r-walk
        l {int} -- length of the walk
        alias_nodes {list} - list with nodes processed probabilities of biased random walk
        alias_edges {list} - list with edges processed probabilities of biased random walk
    Returns:
        walk -- generated biased random walk
    """
    walk = [u]
    for walk_iter in range(l):
        curr = walk[-1]
        neighbors =sorted(n_G.neighbors(curr))
        if len(neighbors) > 0:
            if len(walk) == 1:
                s = neighbors[alias_draw(alias_nodes[curr][0],alias_nodes[curr][1])]
                
            else:
                src = walk[-2]
                s = neighbors[alias_draw(alias_edges[(src,curr)][0], alias_edges[(src,curr)][1])]
            walk.append(s)
        else:
            break
    return walk


def start_weight(G):
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
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
    for nbr in G.neighbors(dst):
        if nbr == src:
            prob = G[dst][nbr]['weight']/p
        elif G.has_edge(src,nbr):
            prob = 1
        else:
            prob = G[dst][nbr]['weight']/q
        probs.append(prob)
    Z = sum(probs) ##normalizing constant
    normalized_probs = [prob/Z for prob in probs]

    return alias_setup(normalized_probs)

