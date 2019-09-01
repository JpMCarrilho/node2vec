import numpy as np
import tensorflow as tf

## todo corpus = 
def pre_encoding(sentences):
    words = []
    for sentence in sentences:
        words.append(word for word in sentence)

    words = set(words)

    word2int = {}
    int2word = {}
    vocab_size = len(words)

    for i,word in enumerate(words):
        word2int[word] = i
        int2word[i] = word
    return word2int, int2word

    data = []
    ## window_size = args 

    for sentence in sentences:
        for word_index, word in enumarate(sentence)
            for nb_word in sentence[max(word_index - window_size,0): min(word_index + window_size,len(sentences))+1])]:
                if nb_word != word:
                      data.append([word,nb_word]) 
    return data,word2int,int2word

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

def word2vec(vocab_size,window_size,embedding_dim,n_iters)
    x_train = []
    y_train = []

    for data_word in data:
        x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
        y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x = tf.placeholder(tf.float32,shape=(None, vocab_size))
    y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

    W1 = tf.Variable(tf.random_normal([vocab_size, embedding_dim]))
    b1 = tf.Variable(tf.random_normal([embedding_dim]))
    hidden = tf.add(tf.matmul(x,W1),b1)

    W2 = tf.Variable(tf.random_normal([embedding_dim, vocab_size]))
    b2 = tf.Variable(tf.random_normal([vocab_size]))

    prediction = tf.nn.sofmax(tf.add(tf.matmul(hidden,W2),b2))

    sess = tf.Session()

    init = tf.global_variables_initializer()

    sess.run(init)

    loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction),reduction_indices = [1]))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    for _ in range(n_iters):
        sess.run(train_step,feed_dict = {x: x_train, y_label:y_train})
        vectors = sess.run(W1 + b1)
        print('loss is: ', sess.run(loss, feed_dict={x:x_train,y_label:y_train}))

    return(vectors)