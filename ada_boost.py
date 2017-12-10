#!/usr/bin/env python2

'''

Following is one of our  idea of implementing AdaBoost:
H = W1 * h1 + W2 * h2 + W3 * h3 + .... + W32 * h32 for each orientation
 
-> Weak Classifiers based on Median value of the feature.
 
1. Initially on the training data we will find the median of all the 192 features for each orientation.
Hence for each feature we will have 4 medians (1 for each orientation)
 
2. Our hypothesis is, the data whose feature is less than the (above calculated) median belongs to that orientation.
Certainly many data would get overlapped (in the sense, same data will be classified to different orientations) - will address
the issue down
 
3. Now for each orientation say 0, we will define a weight metric - intially assigned as 1 / number of train data. Since all weights
are same we will select our first weak orientation from 192 features for that orientation which is giving the highest accuracy.
Accuracy is calculated based on the hypothesis mentioned (<= median as +1, > median as -1). 
W1 is calculated based on log (1 - error) / (error)
 
4. After selecting the first weak orientation for each orientation, the weight matrix is updated by:
Changing the weights of those correctly classified by w <- w * (error)/(1-error) and normalize.
 
5. Using the updated weight matrix, next weak orientation is selected which gives the maximum - sum of the weighted datas that are
correctly classified of the remaining 191 features.
W(i) is again calculated.
 
6. Steps 3-5 are repeated for each orientation.
 
7. Steps 4-5 are repeated for 5 weak classifiers.
 
8. At end of Step 7, model is trained as H = W1 * h1 + W2 * h2 + W3 * h3 + .... + W32 * h32 for each orientation
 
As mentioned 4 H values are calculated and the maxium value of four orientation classes is the predicted orientation.

Some of wrongly classified images
train/10008667845.jpg 270
train/10008667845.jpg 0
train/10008667845.jpg 90
train/10008667845.jpg 180
train/100257121.jpg 90
train/100257121.jpg 180
train/100257121.jpg 270
train/100257121.jpg 0

Our hypothesis is based on median of range of values for each orientation. If train data is different from test data in color range, i.e
if train data consists of all photos taken in day time but test data consists of photos taken in dark. Our weak classifiers fails to
predict appropriate orientation and probably result in accuracy ~ 0.
'''

import numpy as np
import math
import pickle

def readData(input_file):
    with open(input_file) as f:
        data = f.read().split()
        file_data = np.array(data)
        file_data = np.array(np.split(file_data, len(file_data)/194))
        file_names = file_data[:,0]
        file_data = file_data[:,1:].astype(int)
        file_data_labels = file_data[:,0]

    return file_data[:,1:], file_data_labels, file_names

def adaBoost(train_data, train_label, model_file):
    M, N = train_data.shape
    threshold_dict = []

    bag_decision_stumps = { 0: {"selected": [], "weight": np.array([1/float(M)] * M)},\
                            90: {"selected": [], "weight": np.array([1/float(M)] * M)},\
                            180: {"selected": [], "weight": np.array([1/float(M)] * M)},\
                            270: {"selected": [], "weight": np.array([1/float(M)] * M)} }
    for feature_num in range(0, N):
        threshold_dict.append({})
        feature = train_data[:,feature_num]
        feature = feature + np.array([1]*M)
        for orientation in (0, 90, 180, 270):
            class_layer = (train_label == orientation)
            orientation_data = class_layer * feature * 1.0
            mask_layer = np.ma.masked_where(orientation_data == 0, orientation_data)
            threshold_dict[feature_num][orientation] = [np.ma.median(mask_layer)]
            less_than_classified = (feature <= threshold_dict[feature_num][orientation][0])
            less_than_classified = np.invert(np.bitwise_xor(less_than_classified, class_layer))
            accuracy = sum(less_than_classified) / float(M)
            threshold_dict[feature_num][orientation].append(1 if accuracy > 0.5 else -1)
            threshold_dict[feature_num][orientation].append(less_than_classified if accuracy > 0.5 else np.invert(less_than_classified))
            threshold_dict[feature_num][orientation].append(accuracy if accuracy > 0.5 else (1-accuracy))

    # Train model
    for weak_classifier in range(32):
        for orientation in (0, 90, 180, 270):
            (max_weight_data, max_feature_num) = (0, 0)
            mask_layer = (train_label == orientation)
            for feature_num in range(0, N):
                if feature_num not in bag_decision_stumps[orientation]["selected"]:
                    feature = train_data[:,feature_num]
                    feature = feature + np.array([1]*M)
                    stump_data = np.sum(threshold_dict[feature_num][orientation][2] * bag_decision_stumps[orientation]["weight"])
                
                    (max_weight_data, max_feature_num) = max((max_weight_data, max_feature_num), (stump_data, feature_num))

            bag_decision_stumps[orientation]["selected"].append(max_feature_num)
            old_weight = np.invert(threshold_dict[max_feature_num][orientation][2]) * bag_decision_stumps[orientation]["weight"]
            error = sum(old_weight)
            bag_decision_stumps[orientation]["weight"] = threshold_dict[max_feature_num][orientation][2]\
                                                  * bag_decision_stumps[orientation]["weight"] \
                                                  * error \
                                                  / (1 - error)
            bag_decision_stumps[orientation]["weight"] += old_weight
            norm_factor = 1.0 / sum(bag_decision_stumps[orientation]["weight"])
            bag_decision_stumps[orientation]["weight"] *= norm_factor
            threshold_dict[max_feature_num][orientation][3] = error

    with open(model_file, "w") as f:
        pickle.dump([threshold_dict, bag_decision_stumps], f)


def test(test_data, test_label, file_names, model_file):
    with open(model_file, "r") as f:
        threshold_dict, bag_decision_stumps = pickle.load(f)

    # Test data accuracy
    accuracy_count = 0
    with open("output.txt", "wb") as outfile:
        M = len(test_data)
        for row in range(0, len(test_data)):
            vote = {0: 0, 90: 0, 180: 0, 270: 0}
            for orientation in (0, 90, 180, 270):
                value = 0
                for weak_classifier in bag_decision_stumps[orientation]["selected"]:
                    error = threshold_dict[weak_classifier][orientation][3]
                    sign = threshold_dict[weak_classifier][orientation][1]
                    sign *= (1 if (test_data[row][weak_classifier] + 1) <= threshold_dict[weak_classifier][orientation][0] else -1)
                    value += (math.log((1-error)/(error))*sign)
                vote[orientation] = value

            bucket = {0: 0, 90: 0, 180: 0, 270: 0}
            for orientation in (0, 90, 180, 270):
                if vote[orientation] > 0:
                    bucket[orientation] += vote[orientation]
                else:
                    a = [0,90,180,270]
                    a.remove(orientation)
                    for c in a:
                        bucket[c] += abs(vote[orientation])

            predict_label = max(bucket.items(), key = lambda x:x[1])[0]
            if test_label[row] == predict_label:
                accuracy_count += 1
            outfile.write("{} {}\n".format(file_names[row], predict_label))
    print "Accuracy: ", accuracy_count/float(M) * 100

def train(train_file, model_file):
    train_data, train_label, _ = readData(train_file)
    adaBoost(train_data, train_label, model_file)

def predict(test_file, model_file):
    test_data, test_label, file_names = readData(test_file)
    test(test_data, test_label, file_names, model_file)
