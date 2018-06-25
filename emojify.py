import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt

def sentence_to_avg(sentence, word_to_vector_map):
    words = [i.lower() for i in sentence.split()]
    avg = np.zeros((50,))
    for w in words:
        avg += word_to_vector_map[w]
    avg = avg/len(words)
    return avg

def predict(X, Y, W, b, word_to_vector_map):
    np.random.seed(1)
    m = Y.shape[0]
    
    predict = []
    
    """print(Y_oh)
    
    for i in range(m):
        max_index = np.argmax(Y_oh[i])
        actual.append(max_index)
    print(actual)"""
    
    for i in range(m):
        avg = sentence_to_avg(X[i], word_to_vector_map)
        z = np.dot(W, avg)+b
        a = softmax(z)
        
        max_index = np.argmax(a)
        #print(max_index)

        predict.append(max_index)
        
    error_vec = Y - predict
    count = 0
    for i in error_vec:
        if i != 0:
            count += 1
    error = count / m
    
    print("Accuracy ", (1-error))
    
    return predict
            

def model(X, Y, word_to_vector_map, learning_rate, num_iterations):
    np.random.seed(1)
    m = Y.shape[0]
    n_y = 5
    n_h = 50
    
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    Y_oh = convert_to_one_hot(Y, C=n_y)
    
    for t in range(num_iterations):
        for i in range(m):
            avg = sentence_to_avg(X[i], word_to_vector_map)
            z = np.dot(W, avg)+b
            a = softmax(z)
            
            cost = -np.sum(np.multiply(Y_oh[i], np.log(a)))
            
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz
            
            W = W - learning_rate * dW
            b = b - learning_rate * db
            
        if t%100 == 0 :
            print("Epoch: ", t, " cost: ", cost)
            pred = predict(X, Y, W, b, word_to_vector_map)
    return pred, W, b


X_train, Y_train = read_csv('train.csv')
X_test, Y_test = read_csv('test.csv')

index = 1
print(X_train[index], label_to_emoji(Y_train[index]))

Y_oh_train = convert_to_one_hot(Y_train, C=5)
Y_oh_test = convert_to_one_hot(Y_test, C=5)

print(Y_train[index]," is converted to ",Y_oh_train[index])

word_to_index, index_to_word, word_to_vector_map = read_glove_vecs('glove.6B.50d.txt')

word = "cucumber"
index = 289846
print("index of ", word, " is ", word_to_index[word])
print("word at index ", index, " is ", index_to_word[index])
#print("word_to_vector_map", word_to_vector_map["cucumber"])

avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vector_map)
print("avg = ", avg)

pred, W, b = model(X_train, Y_train, word_to_vector_map, 0.015, 4000)
print("Training Dataset")
print("Actual Y_train")
print(Y_train)
print("Predicted Y_train")
print(pred)

pred = predict(X_test, Y_test, W, b, word_to_vector_map)
print("Testing Dataset")
print("Actual Y_test")
print(Y_test)
print("Predicted Y_test")
print(pred)

X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([0, 0,2, 1, 4,3])

pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vector_map)
idx = 0
for s in X_my_sentences:
    print(s, pred[idx])
    idx += 1
#print_predictions(X_my_sentences, pred)