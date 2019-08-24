import numpy as np
import networkx as nx
from random import shuffle
from gensim.models import Word2Vec
import node2vec
import utils

def main():
    #loads graph via edgelist file
    n_G = nx.read_edgelist(path ="karate.edgelist",create_using=nx.DiGraph(),nodetype = int, data =(('weight',float)))
    n_G = n_G.to_undirected()
    
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
        print(nodes)
        for node in nodes:
            walk = n2v.node2vecWalk(node,l)
            walks.append(walk)

    walks = [map(str,walk) for walk in walks]
    model = Word2Vec(walks,size = embedding_dim,window = context_size,iter = 1, workers = 4)
    model.wv.save_word2vec_format("karate2.emd")
        
    if visualize == True:
        vocab = list(model.wv.vocab)
        X = model[vocab]        
        utils.visualize(vocab,X)
if __name__ == '__main__':
        main()