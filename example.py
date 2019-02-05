#!/usr/bin/env python

import knet 
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score


db = {}

db["data_name"]="spiral"
db["data_path"]="./datasets/" + db["data_name"] + "/"
db["orig_data_file_name"]= db["data_path"] + db["data_name"] + ".csv"
db["orig_label_file_name"]= db["data_path"] + db["data_name"] + "_label.csv"
db['train_data_file_name']  = 'datasets/' + db["data_name"] + '/subset_rest/train.csv'
db["train_label_file_name"]= 'datasets/' + db["data_name"] + '/subset_rest/train_label.csv'
db['width_scale'] = 8
db["σ_ratio"]=0.1
db['num_of_clusters'] = 3


#true_label = np.loadtxt(db['data_path'] + db['data_name'] + '_label.csv', delimiter=',', dtype=np.float64)
true_label = np.loadtxt('datasets/' + db["data_name"] + '/subset_rest/train_label.csv', delimiter=',', dtype=np.float64)

[allocation, knet] = knet.cluster(db)
nmi = normalized_mutual_info_score(allocation, true_label)
print('NMI : %.3f'%(nmi))

[a,new_X] = knet(knet.db['train_data'].X_Var)
allocation = KMeans(db['num_of_clusters'], n_init=10).fit_predict(new_X.data.numpy())
nmi = normalized_mutual_info_score(allocation, true_label)
print('NMI : %.3f'%(nmi))



out_of_sample_X = np.loadtxt(db["orig_data_file_name"], delimiter=',', dtype=np.float64)
out_of_sample_true_label = np.loadtxt(db["orig_label_file_name"], delimiter=',', dtype=np.float64)
#out_of_sample_X = np.loadtxt('datasets/cancer/subset_rest/test.csv', delimiter=',', dtype=np.float64)
#out_of_sample_true_label = np.loadtxt('datasets/cancer/subset_rest/test_label.csv', delimiter=',', dtype=np.float64)

out_of_sample_X = torch.tensor(out_of_sample_X)
out_of_sample_X = Variable(out_of_sample_X.type(torch.FloatTensor), requires_grad=False)
[AE_output, new_X] = knet(out_of_sample_X)
allocation = KMeans(db['num_of_clusters'], n_init=10).fit_predict(new_X.data.numpy())
nmi = normalized_mutual_info_score(allocation, out_of_sample_true_label)
print('NMI : %.3f'%(nmi))































#db = {}
#db["data_name"]="cancer"
#db["data_path"]="./datasets/" + db["data_name"] + "/"
#db["orig_data_file_name"]= db["data_path"] + db["data_name"] + ".csv"
#db["orig_label_file_name"]= db["data_path"] + db["data_name"] + "_label.csv"
#db['train_data_file_name']  = 'datasets/cancer/subset_rest/train.csv'
#db["train_label_file_name"]= 'datasets/cancer/subset_rest/train_label.csv'
#db['width_scale'] = 2
#db["σ_ratio"]=1
#db['num_of_clusters'] = 2
#
#
##true_label = np.loadtxt(db['data_path'] + db['data_name'] + '_label.csv', delimiter=',', dtype=np.float64)
#true_label = np.loadtxt('datasets/cancer/subset_rest/train_label.csv', delimiter=',', dtype=np.float64)
#
#[allocation, knet] = knet.cluster(db)
#nmi = normalized_mutual_info_score(allocation, true_label)
#print('NMI : %.3f'%(nmi))
#
#
#out_of_sample_X = np.loadtxt(db["orig_data_file_name"], delimiter=',', dtype=np.float64)
#out_of_sample_true_label = np.loadtxt(db["orig_label_file_name"], delimiter=',', dtype=np.float64)
##out_of_sample_X = np.loadtxt('datasets/cancer/subset_rest/test.csv', delimiter=',', dtype=np.float64)
##out_of_sample_true_label = np.loadtxt('datasets/cancer/subset_rest/test_label.csv', delimiter=',', dtype=np.float64)
#
#out_of_sample_X = torch.tensor(out_of_sample_X)
#out_of_sample_X = Variable(out_of_sample_X.type(torch.FloatTensor), requires_grad=False)
#[AE_output, new_X] = knet(out_of_sample_X)
#allocation = KMeans(db['num_of_clusters'], n_init=10).fit_predict(new_X.data.numpy())
#nmi = normalized_mutual_info_score(allocation, out_of_sample_true_label)
#print('NMI : %.3f'%(nmi))
#
