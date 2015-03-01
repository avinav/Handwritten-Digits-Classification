import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from scipy.io import savemat
from math import sqrt


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
    return  np.reciprocal((1+np.exp(-1*z)))#your code here
    
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
    return data[:,pix_remain]

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

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
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
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
    train_data_matrix = selectfeatures(train_data_matrix,size_train_samples)
    
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
    test_data_matrix = selectfeatures(test_data_matrix,size_test_samples)
    
    #Pick a reasonable size for validation data
    import random
    #rs = random.sample(range(60000),50000)
    #rs_v = list(set(range(60000)) - set(rs))
    rs = random.sample(range(60000),500)
    rs_v = list(set(range(60000)) - set(rs))
    rs_v = random.sample(rs_v,100)
    rs_tt = random.sample(range(test_data_matrix.shape[0]),100)
    tr_data = train_data_matrix[np.array(rs)]
    tr_labels = train_labels[np.array(rs)]
    vs_data = train_data_matrix[np.array(rs_v)]
    vs_labels = train_labels[np.array(rs_v)]    
    tt_data = test_data_matrix[np.array(rs_tt)]
    tt_labels = test_labels[np.array(rs_tt)]
    #Your code here
    #train_data = np.array([])
    #train_label = np.array([])
    #validation_data = np.array([])
    #validation_label = np.array([])
    #test_data = np.array([])
    #test_label = np.array([])
    #return tr_data, tr_labels, vs_data, vs_labels, test_data_matrix, test_labels
    #return train_data, train_label, validation_data, validation_label, test_data, test_label

    return tr_data, tr_labels, vs_data, vs_labels, tt_data, tt_labels    
    
    

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
    
    #Your code here
    #
    #
    #
    #
    #
    n_samples = training_data.shape[0];
    inp = np.array([0]*(n_input+1),dtype='float32')
    hid = np.array([0]*(n_hidden+1),dtype='float32')
    out = np.array([0]*n_class,dtype='float32')
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    gradw1sum = np.zeros(w1.shape)
    gradw2sum = np.zeros(w2.shape)
    obj_val = 0
    lab = np.array([0]*10)
    for sam in range(n_samples):
        inp = np.concatenate((training_data[sam,:],[1])) #appending bias terms
        hid[n_hidden] = 1 #setting bias terms
        lab = training_label[sam,:]
        # feed forward
        for k in range(n_hidden):
            for i in range(n_input+1):
                hid[k] += inp[i]*w1[k][i]
            hid[k] = sigmoid(hid[k])
        for l in range(n_class):
            for k in range(n_hidden+1):
                out[l] += hid[l]*w2[l][k]
            out[l] = sigmoid(out[l])
        # back propagation
        for l  in range(n_class):
            obj_val += lab[l]*np.log(out[l]) + (1-lab[l])*np.log(1 - out[l])
        delval = out - lab
        
        for l in range(n_class):
            for k in range(n_hidden+1):
                grad_w2[l][k] = delval[l]*hid[k]
        
        for k in range(n_hidden):
            for i in range(n_input+1):
                temp = 0;
                for l in range(n_class):
                    temp += delval[l]*w2[l][k]
                grad_w1[l][k] = (1 - hid[k])*hid[k]*temp*inp[i]
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
    #obj_grad = np.array([])
    obj_val = obj_val/n_samples
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
    
    #labels = np.array([])
    #Your code here
    n_input = w1.shape[1] - 1
    n_hidden = w1.shape[0]
    n_class = w2.shape[0]
    n_samples = data.shape[0];
    inp = np.array([0]*(n_input+1),dtype='float32')
    hid = np.array([0]*(n_hidden+1),dtype='float32')
    out = np.array([0]*(n_class),dtype='float32')
    label = np.array([])
    for sam in range(n_samples):
        inp = np.concatenate((data[sam,:],[1])) #appending bias terms
        hid[n_hidden] = 1 #setting bias terms
        # feed forward
        for k in range(n_hidden):
            for i in range(n_input+1):
                hid[k] += inp[i]*w1[k][i]
            hid[k] = sigmoid(hid[k])
        for l in range(n_class):
            for k in range(n_hidden+1):
                out[l] += hid[l]*w2[l][k]
            out[l] = sigmoid(out[l])
        label = np.append(label,np.argmax(out))
    return label



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


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

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

w_dict = {'w1': w1,'w2' : w2}
savemat('weights.mat',w_dict)
#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')