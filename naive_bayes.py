import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
from pylab import savefig,cm
def preprocess_cpt(feat,labels,n_class):
    """% Take input as out matrix n_samples*n_nodes and labels n_samples*n_nodes
    % Input: 
    % n_class = [0,..9]
    % Output: 
    % out_dict
    % out: dictionary
    """
    feat[feat>=0.3] = 1
    feat[feat<0.3] = 0
    feat = feat.astype('int')
    #labels = np.argmax(labels,1)
    feat_dict = {}
    for l in n_class:
        feat_dict[str(l)] = feat[labels == l]
    return feat_dict

def construct_cpt(data, n_features,vals,labels):
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

    cpt = {}
    for l in labels:
        class_data = data[str(l)]
        feature_matrix = np.ones((n_features,vals.size))
        for sam in range(class_data.shape[0]):
            for node in range(n_features):
                feature_matrix[node][class_data[sam][node]] += 1
        feature_matrix = feature_matrix/np.sum(feature_matrix,1).reshape(feature_matrix.shape[0],1)
        cpt[str(l)] = feature_matrix
    return cpt

def naive_bayes(cpt,test_data,labels):
    #test_data = test_data/np.sum(test_data,1).reshape(test_data.shape[0],1)
    #test_data = test_data*10
    #test_data = test_data.astype('int')
    test_labels = np.array([])
    for sam in range(test_data.shape[0]):
        logprob = np.zeros(labels.size)
        data = test_data[sam]
        for l in labels:
            feature_matrix = cpt[str(l)]
            for j in range(data.size):
                logprob[l] += np.log(feature_matrix[j][data[j]])
        test_labels = np.append(test_labels,np.argmax(logprob))        
    return test_labels
        
    
'''******************************Naive Bayes**************************'''
mat = loadmat('/home/avinav/ML/mnist_reduced_sdev.mat')
train_data = {}
train_labels = np.array([])
train_data_matrix = np.array([],dtype='int')
for i in range(0,10):
    im_data = mat.get('train'+str(i))
    im_data = im_data.astype('float32')
    im_data = im_data/255
    im_data[im_data>=0.3] = 1
    im_data[im_data<0.3] = 0    
    train_data[str(i)] = im_data.astype('int')
    train_labels = np.append(train_labels,[i]*im_data.shape[0])
    if (i==0):
        train_data_matrix = im_data.astype('int')
    else:
        train_data_matrix = np.concatenate((train_data_matrix,im_data.astype('int')))

test_data = {}
test_labels = np.array([])
test_data_matrix = np.array([],dtype='int')
for i in range(0,10):
    im_data = mat.get('test'+str(i))
    im_data = im_data.astype('float32')
    im_data = im_data/255
    im_data[im_data>=0.3] = 1
    im_data[im_data<0.3] = 0    
    test_data[str(i)] = im_data.astype('int') 
    test_labels = np.append(test_labels,[i]*im_data.shape[0])
    if (i==0):
        test_data_matrix = im_data.astype('int')
    else:
        test_data_matrix = np.concatenate((test_data_matrix,im_data.astype('int')))
n_features = train_data[str(0)].shape[1]
vals = np.array(np.linspace(0,1,num=2)).astype('int')
labels = np.array(np.linspace(0,9,num=10)).astype('int')

cpt = construct_cpt(train_data, n_features,vals,labels)
savemat('cpt_navie_bayes_feature.mat',cpt)
#cpt = loadmat('cpt_navie_bayes.mat')
train_pred_labels = naive_bayes(cpt,train_data_matrix,labels)
test_pred_labels = naive_bayes(cpt,test_data_matrix,labels)

print('\n Train set Accuracy:' + str(100*np.mean(train_pred_labels == train_labels)) + '%')
print('\n Test set Accuracy:' + str(100*np.mean(test_pred_labels == test_labels)) + '%')



conf_matrix = np.zeros((10,10))
conf_matrix_string = np.zeros((10,10))
num_digits = np.zeros(10)
for i in range(10):
    for j in range(10):
        conf_matrix[i][j] = 100*np.mean(test_pred_labels[test_labels == i] == j)
        conf_matrix_string[i][j] = "%.2f" % conf_matrix[i][j]
    num_digits[i] = len(test_pred_labels[test_labels == i])
conf_matrix = conf_matrix.tolist()        
savemat('conf_matrix_naive_bayes.mat',{'conf_matrix':conf_matrix,'conf_matrix_string':conf_matrix_string, 'num_digits': num_digits})



#norm_conf = np.zeros((10,10))
#norm_conf_string = np.zeros((10,10))
#for i in range(10):
#    for j in range(10):
#        norm_conf[i][j] = 100*np.mean(test_pred_labels[test_labels == i] == j)
#        norm_conf_string[i][j] = "%.2f" % norm_conf[i][j]
#            
#norm_conf = norm_conf.tolist()
#
#plt.clf()
#fig = plt.figure()
#ax = fig.add_subplot(111)
#res = ax.imshow(np.array(norm_conf), cmap=cm.jet, interpolation='nearest')
#for i, cas in enumerate(norm_conf_string):
#    for j, c in enumerate(cas):
#        if c>0:
#            plt.text(j-.3, i+.3, c, fontsize=12)
#cb = fig.colorbar(res)
#savefig("naive_bayes_feature_confusion.png", format="png")
#fig.show()