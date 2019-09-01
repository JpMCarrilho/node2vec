import numpy as np
import networkx as nx
from random import shuffle
from gensim.models import Word2Vec
import node2vec
import utils
import word2vec

def main():
    #loads graph via edgelist file
    n_G = nx.read_edgelist(path ="karate.edgelist",create_using=nx.DiGraph(),nodetype = int, data =(('weight',float)))
    n_G = n_G.to_undirected()
    n_iters = 1000 
    l = 80 #length of random walk
    n_walks = 30 #number of random walks
    context_size = 10
    embedding_dim  = 2
    p = 0.5
    q = 1
    walks = []
    nodes = list(n_G.nodes())
    visualize = True
    
    n2v = node2vec.node2vec(n_G,p,q,l)
    
    for r in range(n_walks):
        print("Walk " + str(r+1))
        shuffle(nodes)
        
        for node in nodes:
            walk = n2v.node2vecWalk(node,l)
            walks.append(walk)

    walks = [map(str,walk) for walk in walks]

    vectors,word2int = word2vec.model(sentences = walks,embedding_dim = embedding_dim,window_size = context_size,n_iters = n_iters)
    print(word2int)
    print(vectors)
    if visualize == True:
        
        utils.visualize(word2int,vectors)
if __name__ == '__main__':
        main()