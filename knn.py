import numpy as np
import math
from collections import Counter

'''
Implements K nearest neighbour classifier. During the training phase we just dump the train
data to model_file.txt. The classifier predicts the orientaion of the image by looking at the orientaion of k closest neighbours
and assigns the orientaion which has maximum count. The plot consiting of variations of accuracies of the model
with K values from 3 to 50 is shown in knn_plot.png. The maximum accuray of 71.63 was obtained at K = 43.For overfitting
reasons we have restricted k to 11 which gives an accuray of 71.04%
'''

class KNN():

    #constructor
    def __init__(self):
        self.train_features_vec = []
        self.test_features_vec = []
        self.correct_count = 0
        self.accuray = 0
        self.k = 11

    """
    calculates euclidian distance
    input: train_dataset,an instance of test data
    return : vector of euclidian distances between test data and train data
    """
    def calculate_euclidian_distance(self,X_train,X_test):
        #return [math.sqrt(sum(row**2))for row in X_train - X_test]
        return np.sqrt(np.sum(np.subtract(X_train,X_test)**2,axis =1))
    """
    dumps the train data to knn_file.txt
    """
    def train(self,train_fname,model_file):
        with open(train_fname) as f:
            for line in f.readlines():
                elements = line.split(" ")
                self.train_features_vec.append(elements[1:])

        self.train_features_vec_np = np.array(self.train_features_vec,dtype = int)
        #print(self.train_features_vec_np)
        np.savetxt(model_file,self.train_features_vec_np,fmt='%i',delimiter = ' ')

    """
    predicts the orientaion of the image by looking at the orientaion of k closest neighbours
    and assigns the orientaion which has maximum count.
    """
    def predict(self,test_fname,model_file):
        image_id = []
        output_str = ''
        with open(test_fname) as f:
            for line in f.readlines():
                elements = line.split(" ")
                image_id.append(elements[0])
                self.test_features_vec.append(elements[1:])
                #print(map(int,elements[1:]))

        self.test_features_vec_np = np.array(self.test_features_vec,dtype = int)
        self.train_features_vec_np = np.loadtxt(model_file,delimiter = ' ')
        #self.train_features_vec_np = self.train_features_vec_np.astype(int)
        #print(self.train_features_vec_np.dtype)

        train_data,train_label = self.train_features_vec_np[:,1:],self.train_features_vec_np[:,0]
        test_data,test_label = self.test_features_vec_np[:,1:],self.test_features_vec_np[:,0]

        for index,x_test in enumerate(test_data):
            euc_distance_vec = self.calculate_euclidian_distance(train_data,x_test)
            eucdist_distance_vec = np.column_stack((euc_distance_vec,train_label))
            knn_vec = eucdist_distance_vec[np.argpartition(eucdist_distance_vec[:,0],self.k)][0:self.k,1]
            knn_vec_counter = Counter(knn_vec)
            predict_label = knn_vec_counter.most_common(1)[0][0]
            output_str += image_id[index] + ' ' + str(int(predict_label)) + '\n'

            if(predict_label == test_label[index]):
                self.correct_count +=1
        print("K = %d Accuray = %f"%(self.k,((float(self.correct_count)/len(test_label))*100)))
        with open('output.txt','w') as f:
            f.write(output_str)

