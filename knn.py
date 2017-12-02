import numpy as np
import math
from collections import Counter
import time

train_features_vec = []
test_features_vec = []
accuray_count = 0

def calculate_euclidian_distance(X_train,X_test):
    #return [math.sqrt(sum(row**2))for row in X_train - X_test]
    return np.sqrt(np.sum(np.subtract(X_train,X_test)**2,axis =1))


with open('train-data.txt') as f:
    for line in f.readlines():
        elements = line.split(" ")
        train_features_vec.append(map(int,elements[1:]))
        #print(map(int,elements[1:]))

train_features_vec_np = np.array(train_features_vec)
#print(train_features_vec_np)

with open('test-data.txt') as f:
    for line in f.readlines():
        elements = line.split(" ")
        test_features_vec.append(map(int,elements[1:]))
        #print(map(int,elements[1:]))

test_features_vec_np = np.array(test_features_vec)

train_data,train_label = train_features_vec_np[:,1:],train_features_vec_np[:,0]
test_data,test_label = test_features_vec_np[:,1:],test_features_vec_np[:,0]
#start_time = time.time()
for index,x_test in enumerate(test_data):
    #start_time = time.time()
    euc_distance_vec = calculate_euclidian_distance(train_data,x_test)
    #end_time = time.time()
    #print("calculate_euclidian_distance %f"%(end_time - start_time))
    #print(len(euc_distance_vec))
    #start_time = time.time()
    #eucdist_set = [item for item in zip(train_label,euc_distance_vec)]
    #print(train_label.shape)
    #print(euc_distance_vec.shape)
    eucdist_distance_vec = np.column_stack((euc_distance_vec,train_label))
    #end_time = time.time()
    #print("eucdist_set %f"%(end_time - start_time))
    #print(eucdist_set)
    #knn_vec = sorted(eucdist_distance_vec,key=lambda row:row[0])
    #print(knn_vec)
    #knn_vec = eucdist_distance_vec[eucdist_distance_vec[:,0].argsort()][0:11,1]
    #print(eucdist_distance_vec.shape)
    knn_vec = eucdist_distance_vec[np.argpartition(eucdist_distance_vec[:,0],11)][0:11,1]
    #print(knn_vec[0:9,1])
    #print(knn_vec)
    #knn_vec_label = [item[1]for item in knn_vec]
    knn_vec_counter = Counter(knn_vec)
    predict_label = knn_vec_counter.most_common(1)[0][0]
    if(predict_label == test_label[index]):
        accuray_count +=1
    #print(index,predict_label,test_label[index])
    #print("***********************************************************************")
#print(accuray_count)
#print(len(test_label))
#end_time = time.time()
#print(start_time - end_time)
print("Accuray = %f"%(float(accuray_count)/len(test_label)))

#print(train_data)
#print(test_data)
#print(train_label)
#print(test_label)
