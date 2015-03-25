import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from scipy.io import savemat
from math import sqrt
_itr = 0

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  np.reciprocal((1+np.exp(-1*z)))
    
def selectfeatures(orig_data,size):
    import random
    data = np.copy(orig_data)
    i = 0
    n = 50
    samples = {}
    lab = 0
    for j in size:
        s = random.sample(range(i,i+j),n)
        samples[str(lab)] = data[s]
        #samples[str(lab)][samples[str(lab)]<0.5] = 0
        #samples[str(lab)][samples[str(lab)]>=0.5] = 1
        i = j 
        lab = lab + 1
    #count0 = np.array([0]*10)
    pixel = np.array([-1]*10)
    mark_remove = np.array([],dtype='int32')
    for j in range(data.shape[1]):
        for l in range(lab):
            sam = samples[str(l)]
            count0 = sam[:,j][sam[:,j]<0.05].size
            if(count0 < n/10):
                pixel[l] = 1
            elif(count0 > 8*n/10):
                pixel[l] = 0
        if(pixel[pixel==1].size == lab or pixel[pixel==0].size == lab):
            mark_remove = np.append(mark_remove,j)
    
    pix_remain = np.array(list(set(range(data.shape[1])) - set(mark_remove)))
    return data[:,pix_remain],pix_remain

def preprocess():
    """ Input:
    load the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('/home/avinav/ML/basecode/mnist_all.mat') #loads the MAT object as a Dictionary
    im_data = mat.get('train0')
    im_data = im_data.astype('float32')
    train_data_matrix = im_data/255
    lab = np.array([0]*10)
    lab[0] = 1
    train_labels = np.array([lab]*im_data.shape[0])
    size_train_samples = np.array([0]*10)
    size_train_samples[0] = im_data.shape[0]
    for i in range(1,10):
        im_data = mat.get('train'+str(i))
        im_data = im_data.astype('float32')
        im_data = im_data/255
        train_data_matrix = np.concatenate((train_data_matrix,im_data))
        #train_labels = np.concatenate((train_labels,np.array([i]*im_data.shape[0])))
        lab = np.array([0]*10)
        lab[i] = 1
        lab = np.array([lab]*im_data.shape[0])
        size_train_samples[i] = im_data.shape[0]
        train_labels = np.vstack((train_labels,lab))
    #train_data_matrix,pix_remain = selectfeatures(train_data_matrix,size_train_samples)
    
    im_data = mat.get('test0')
    im_data = im_data.astype('float32')
    test_data_matrix = im_data/255
    lab = np.array([0]*10)
    lab[0] = 1
    test_labels = np.array([lab]*im_data.shape[0])
    size_test_samples = np.array([0]*10)
    size_test_samples[0] = im_data.shape[0]
    for i in range(1,10):
        im_data = mat.get('test'+str(i))
        im_data = im_data.astype('float32')
        im_data = im_data/255
        test_data_matrix = np.concatenate((test_data_matrix,im_data))
        #test_labels = np.concatenate((test_labels,np.array([i]*im_data.shape[0])))
        lab = np.array([0]*10)
        lab[i] = 1
        lab = np.array([lab]*im_data.shape[0])
        size_test_samples[i] = im_data.shape[0]
        test_labels = np.vstack((test_labels,lab))
    #Pick a reasonable size for validation data
    import random
    rs = random.sample(range(60000),50000)
    rs_v = list(set(range(60000)) - set(rs))

    tr_data = train_data_matrix[np.array(rs)]
    tr_labels = train_labels[np.array(rs)]
    vs_data = train_data_matrix[np.array(rs_v)]
    vs_labels = train_labels[np.array(rs_v)]    
    return tr_data, tr_labels, vs_data, vs_labels, test_data_matrix, test_labels    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    n_samples = training_data.shape[0];
    inp = np.zeros((1,n_input+1))
    hid = np.zeros((1,n_hidden+1))
    out = np.zeros((1,n_class))
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    gradw1sum = np.zeros(w1.shape)
    gradw2sum = np.zeros(w2.shape)
    obj_val = 0
    lab = np.array([0]*10)
    hid[0][n_hidden] = 1 #setting bias terms
    for sam in range(n_samples):
        inp = np.concatenate((training_data[sam,:],[1])) #appending bias terms
        lab = training_label[sam,:]
        # feed forward
        inp = inp.reshape(1,inp.size)
        hid[0][:-1] = np.dot(inp,w1.T)
        hid[0][:-1] = sigmoid(hid[0][:-1])
        out = np.dot(hid,w2.T)
        out = sigmoid(out)
        # back propagation
        obj_val_temp = lab*np.log(out) + (1-lab)*np.log(1 - out)
        obj_val += np.sum(obj_val_temp)
        delval = out - lab
        
        grad_w2 = np.dot(delval.T,hid)
        temp = np.zeros((1,n_hidden))
        temp = np.dot(delval,w2)
        temp1 = ((1-hid[0][:-1])*hid[0][:-1]*temp[0][:-1])
        temp1 = temp1.reshape(1,temp1.size)
        grad_w1 = np.dot(temp1.T,inp)
        gradw1sum += grad_w1
        gradw2sum += grad_w2
    
    w1sum = np.sum(np.square(w1))
    w2sum = np.sum(np.square(w2))
    obj_val += lambdaval*(w1sum + w2sum)/2
    
    grad_w1 = (gradw1sum + lambdaval*w1)/n_samples
    grad_w2 = (gradw2sum + lambdaval*w2)/n_samples    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_val = -1*obj_val/n_samples
    return (obj_val,obj_grad)

def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    n_input = w1.shape[1] - 1
    n_hidden = w1.shape[0]
    n_class = w2.shape[0]
    n_samples = data.shape[0];
    inp = np.zeros((1,n_input+1))
    hid = np.zeros((1,n_hidden+1))
    out = np.zeros((1,n_class))
    label = np.array([])
    hid[0][n_hidden] = 1 #setting bias terms
    initLabel = False
    out_data = np.array([])
    for sam in range(n_samples):
        inp = np.concatenate((data[sam,:],[1])) #appending bias terms
        inp = inp.reshape(1,inp.size)
        # feed forward
        hid[0][:-1] = np.dot(inp,w1.T)
        hid[0][:-1] = sigmoid(hid[0][:-1])
        out = np.dot(hid,w2.T)
        out_data = np.append(out_data,out)
        out = sigmoid(out)
        
        temp = np.array([0]*10)
        temp[np.argmax(out)] = 1
        if (initLabel):
            label = np.vstack((label,temp))
        else:
            label = np.copy(temp)
            initLabel = True
    return label,out_data.reshape(n_samples,out.size)

def preprocess_test_data(test_data):
    test_data = test_data*10
    test_data = test_data.astype('int')
    return test_data

def preprocess_cpt(out,labels,n_class):
    """% Take input as out matrix n_samples*n_nodes and labels n_samples*n_nodes
    % Input: 
    % n_class = [0,..9]
    % Output: 
    % out_dict
    % out: dictionary
    """
    min_out = np.min(out,1)
    min_out = min_out.reshape(min_out.size,1)
    min_out = np.absolute(min_out)
    out += min_out
    
    out = out/np.sum(out,1).reshape(out.shape[0],1)
    out = out*10
    out = out.astype('int')
    labels = np.argmax(labels,1)
    out_dict = {}
    for l in n_class:
        out_dict[str(l)] = out[labels == l]
    return out_dict

def construct_cpt(out, n_nodes):
    """% out is a dictionary, key = class, value = discritized, converted from (0,1) to [0,1,..9] for
    % simplifying indexing of feature matrix. 
    % feature_matrix[i][j] counts the number of times i-th output node takes value as j.
    % cpt {'class-label(c)': feature_matrix[i][j] } can be read as P(F(i)=j|C=c)
    %Input:
    % out is a dictionary, key = class label, value = discritized, converted from (0,1) to [0,1,..9]
    % values of output node
    % n_nodes = number of output nodes
    % Output:
    % cpt = dictionary{'class label': feature_matrix},
    % where feature_matrix = out_nodes*attribute-values 
    """
    
    n_vals = np.array(np.linspace(0,9,num=10),dtype='int')
    labels =  np.array(np.linspace(0,9,num=10),dtype='int')
    cpt = {}
    for l in labels:
        out_data = out[str(l)]
        feature_matrix = np.ones((n_nodes,n_vals.size))
        for sam in range(out_data.shape[0]):
            for node in range(n_nodes):
                feature_matrix[node][out_data[sam][node]] += 1
        feature_matrix = feature_matrix/np.sum(feature_matrix,1).reshape(feature_matrix.shape[0],1)
        cpt[str(l)] = feature_matrix
    return cpt

def naive_bayes(cpt,test_data,n_class):
    min_test_data = np.min(test_data,1)
    min_test_data = min_test_data.reshape(min_test_data.size,1)
    min_test_data = np.absolute(min_test_data)
    test_data += min_test_data
    
    test_data = test_data/np.sum(test_data,1).reshape(test_data.shape[0],1)
    test_data = test_data*10
    test_data = test_data.astype('int')
    test_labels = np.array([])
    for sam in range(test_data.shape[0]):
        prob = np.zeros(n_class.size)
        data = test_data[sam]
        for l in n_class:
            feature_matrix = cpt[str(l)]
            for j in range(data.size):
                prob[l] += np.log(feature_matrix[j][data[j]])
        test_labels = np.append(test_labels,np.argmax(prob))        
    return test_labels
        
    
"""**************Neural Network Script Starts here********************************"""
import time
start_time = time.time()
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
print ("pre-processed!")

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter':50}    # Preferred value.

#Reshape nnParams from 1D vector into w1 and w2 matrices
#w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
#w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#w_dict = {'w1': w1,'w2' : w2}
#savemat('weights.mat',w_dict)
w_dict = loadmat('weights.mat')
w1 = w_dict['w1']
w2 = w_dict['w2']


#Test the computed parameters

class_list = np.array(range(0,n_class))
predicted_label,train_out_data = nnPredict(w1,w2,train_data)
train_out_dict = preprocess_cpt(train_out_data,train_label,class_list)
cpt = construct_cpt(train_out_dict,10)
savemat('cpt_out.mat',cpt)
#cpt=loadmat('cpt_out.mat')
#find the accuracy on Training Dataset
bayes_label = naive_bayes(cpt,train_out_data,class_list)
print('\n Training set Accuracy:' + str(100*np.mean(np.argmax(predicted_label,1) == np.argmax(train_label,1))) + '%')
train_label_digit = np.argmax(train_label,1)
print('\n Training set Accuracy using Bayes:'+ str(100*np.mean(bayes_label == np.argmax(train_label,1))) + '%')
for i in range(10):
    print('True positives, '+ str(i) +': '+
    str(100*np.mean(bayes_label[train_label_digit==i] == train_label_digit[train_label_digit==i])) + '%')

predicted_label,valid_out_data = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset
bayes_label = naive_bayes(cpt,valid_out_data,class_list)
print('\n Validation set Accuracy:' + str(100*np.mean(np.argmax(predicted_label,1) == np.argmax(validation_label,1))) + '%')
validation_label_digit = np.argmax(validation_label,1)
print('\n Validation set Accuracy using Bayes:'+ str(100*np.mean(bayes_label == np.argmax(validation_label,1))) + '%')
for i in range(10):
    print('True Positives, '+ str(i) +': '+
    str(100*np.mean(bayes_label[validation_label_digit ==i] == validation_label_digit[validation_label_digit == i])) + '%')

predicted_label,test_out_data = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
bayes_label = naive_bayes(cpt,test_out_data,class_list)
print('\n Test set Accuracy:' + str(100*np.mean(np.argmax(predicted_label,1) == np.argmax(test_label,1))) + '%')
test_label_digit = np.argmax(test_label,1)
print('\n Test set Accuracy using Bayes:' + str(100*np.mean(bayes_label == np.argmax(test_label,1))) + '%')

for i in range(10):
    print('True Positives, '+ str(i) +': '+
    str(100*np.mean(bayes_label[test_label_digit ==i] == test_label_digit[test_label_digit == i])) + '%')

conf_matrix = np.zeros((10,10))
conf_matrix_string = np.zeros((10,10))
num_digits = np.zeros(10)
for i in range(10):
    for j in range(10):
        conf_matrix[i][j] = 100*np.mean(bayes_label[test_label_digit == i] == j)
        conf_matrix_string[i][j] = "%.2f" % conf_matrix[i][j]
    num_digits[i] = len(test_label_digit[test_label_digit == i])
conf_matrix = conf_matrix.tolist()        
savemat('conf_matrix_nn_bayes_out.mat',{'conf_matrix':conf_matrix,'conf_matrix_string':conf_matrix_string, 'num_digits': num_digits})

print('Time taken %s seconds'% (time.time() - start_time))