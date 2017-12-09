from __future__ import division
import numpy as np
import operator
import matplotlib.pyplot as plt
import math
import operator

'''
Model:

Each image is divided into four corners top,right,bottom,left and we consider 3 first/last rows/colums in each of these corners.

Assumptions:
1. In a 0 degree rotated image, the blue pixels values at the top of the image will have higher values.
2. In a 0 degree rotated image, the red pixels values at the bottom of the image will have lower values.(Brown is considered as darker shade of red)
3. In a 0 degree rotated image, the green pixels values at the bottom of the image will have lower values.( Grass is considered as darker shade of green)

Similar assumptions have been extended for 90,180,270 degree rotated images

Our hypothesis consist of 12 weak decision stumps, 3 for each of the 4 image orientation:
h1 - checks for red pixel values at the bottom of the image and if it is the minimum
     among all the red values at the four corners of the images it will classify the image as 0 otherwise not 0
h2 - checks for green pixel values at the bottom of the image and if it is the minimum
     among all the green values at the four corners of the images it will classify the image as 0 otherwise not 0
h3 - checks for blue pixel values at the top of the image and if it is the maximum
     among all the red values at the four corners of the images it will classify the image as 0 otherwise not 0
h4 - checks for red pixel values at the left of the image and if it is the minimum
     among all the red values at the four corners of the images it will classify the image as 90 otherwise not 90
h5 - checks for green pixel values at the left of the image and if it is the minimum
     among all the green values at the four corners of the images it will classify the image as 90 otherwise not 90
h6 - checks for blue pixel values at the right of the image and if it is the maximum
     among all the red values at the four corners of the images it will classify the image as 90 otherwise not 90
h7 - checks for red pixel values at the top of the image and if it is the minimum
     among all the red values at the four corners of the images it will classify the image as 180 otherwise not 180
h8 - checks for green pixel values at the top of the image and if it is the minimum
     among all the green values at the four corners of the images it will classify the image as 180 otherwise not 180
h9 - checks for blue pixel values at the bottom of the image and if it is the maximum
     among all the red values at the four corners of the images it will classify the image as 180 otherwise not 180
h10 - checks for red pixel values at the right of the image and if it is the minimum
     among all the red values at the four corners of the images it will classify the image as 270 otherwise not 270
h11 - checks for green pixel values at the right of the image and if it is the minimum
     among all the green values at the four corners of the images it will classify the image as 270 otherwise not 270
h12 - checks for blue pixel values at the left of the image and if it is the maximum
     among all the red values at the four corners of the images it will classify the image as 270 otherwise not 270

Prediction:
We take a weighted vote of all these hypothesis. For example if decision stump h1 says that image has a 0 degree
orientation and h10 says the image has a 270 degree orientation then we assign the oritentation based on th weights
of h1 and h10 decision stumps.

Some of wrongly classified images
test/8412266723.jpg
test/9297964944.jpg
test/9466725145.jpg
test/8013919722.jpg
test/8186083991.jpg

Our assumption about the images might not work well on those images consists of rocks/mountains.Some of the images have
might not have green or brown in the bottom. For exmaple images consisting of a road at the bottom will be grey
'''


N = 0
Z = {}



def calculate_binary_label(train_label_np,H):
    labels_list = {}
    for k,v in H.iteritems():
        labels_list[k] = []
        if(v == 0):
            for i in range(0,N):
                if(train_label_np[i] == 0):
                    labels_list[k].append(1)
                else:
                    labels_list[k].append(0)
        elif v == 90:
            for i in range(0,N):
                if(train_label_np[i] == 90):
                    labels_list[k].append(1)
                else:
                    labels_list[k].append(0)
        elif v == 180:
            for i in range(0,N):
                if(train_label_np[i] == 180):
                    labels_list[k].append(1)
                else:
                    labels_list[k].append(0)
        else:
            for i in range(0,N):
                if(train_label_np[i] == 270):
                    labels_list[k].append(1)
                else:
                    labels_list[k].append(0)

    return  labels_list


def calculate_error(train_data_np,binary_label,H,W):
    global N
    #global W

    error_list = {}
    Weight_vec = {}
    train_predict_label_hypo = {}
    for h in H.keys():
        myerror = 0
        error = 0
        train_predict_label = h(train_data_np,N)
        train_predict_label_np = np.array(train_predict_label)
        binary_train_np = np.array(binary_label[h])
        for i in range(0,N):
            if(train_predict_label_np[i] != binary_train_np[i]):
                error = error + W[i]
                myerror +=1
        error_list[h] = error
    h,min_error =  min(error_list.iteritems(), key=operator.itemgetter(1))
    for i in range(0,N):
        if(train_predict_label_np[i] == binary_label[h][i]):
            W[i] = W[i]*((min_error)/(1-min_error))
    W = Normalize(W)
    del H[h]
    del binary_label[h]
    return h,min_error,W

def AdaBoost(train_data_np,train_label_np,H):
    global N
    W = [1/N]*N
    binary_label_hypos = calculate_binary_label(train_label_np,H)
    while(len(H)):
        h,error,W = calculate_error(train_data_np,binary_label_hypos,H,W)
        Z[h] = math.log((1-error)/error)

def Normalize(W):
    s =sum(W)
    return [i/s for i in W ]


def h1(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,0])
        bsum_right = np.sum(train_features_vec_np[:,-n:,0])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,0])
        bsum_left = np.sum(train_features_vec_np[:,:n,0])

        if(bsum_bottom == min(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec

def h2(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,1])
        bsum_right = np.sum(train_features_vec_np[:,-n:,1])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,1])
        bsum_left = np.sum(train_features_vec_np[:,:n,1])

        if(bsum_bottom == min(bsum_top,bsum_right,bsum_bottom,bsum_left)): #0
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec

def h3(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,2])
        bsum_right = np.sum(train_features_vec_np[:,-n:,2])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,2])
        bsum_left = np.sum(train_features_vec_np[:,:n,2])

        if(bsum_top == max(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec

def h4(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,0])
        bsum_right = np.sum(train_features_vec_np[:,-n:,0])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,0])
        bsum_left = np.sum(train_features_vec_np[:,:n,0])

        #max_dict = {180:bsum_0,270:bsum_90,0:bsum_180,90:bsum_270}
        #predict_label =max(max_dict.iteritems(), key=operator.itemgetter(1))[0]
        if(bsum_left == min(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec

def h5(train_data_np,N):

    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,1])
        bsum_right = np.sum(train_features_vec_np[:,-n:,1])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,1])
        bsum_left = np.sum(train_features_vec_np[:,:n,1])

        #max_dict = {180:bsum_0,270:bsum_90,0:bsum_180,90:bsum_270}
        #predict_label =max(max_dict.iteritems(), key=operator.itemgetter(1))[0]
        if(bsum_left == min(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)

    return predict_label_vec

def h6(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,2])
        bsum_right = np.sum(train_features_vec_np[:,-n:,2])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,2])
        bsum_left = np.sum(train_features_vec_np[:,:n,2 ])


        #max_dict = {0:bsum_0,90:bsum_90,180:bsum_180,270:bsum_270}
        #predict_label =max(max_dict.iteritems(), key=operator.itemgetter(1))[0]
        if(bsum_right == max(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec

def h7(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,0])
        bsum_right = np.sum(train_features_vec_np[:,-n:,0])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,0])
        bsum_left = np.sum(train_features_vec_np[:,:n,0])

        #max_dict = {180:bsum_0,270:bsum_90,0:bsum_180,90:bsum_270}
        #predict_label =max(max_dict.iteritems(), key=operator.itemgetter(1))[0]
        if(bsum_top == min(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec


def h8(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,1])
        bsum_right = np.sum(train_features_vec_np[:,-n:,1])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,1])
        bsum_left = np.sum(train_features_vec_np[:,:n,1])

        #max_dict = {180:bsum_180,270:bsum_270,0:bsum_0,90:bsum_90}
        #predict_label =max(max_dict.iteritems(), key=operator.itemgetter(1))[0]
        if(bsum_top == min(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec


def h9(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,2])
        bsum_right = np.sum(train_features_vec_np[:,-n:,2])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,2])
        bsum_left = np.sum(train_features_vec_np[:,:n,2])

        #max_dict = {0:bsum_0,90:bsum_90,180:bsum_180,270:bsum_270}
        #predict_label =max(max_dict.iteritems(), key=operator.itemgetter(1))[0]
        if(bsum_bottom == max(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec

def h10(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,0])
        bsum_right = np.sum(train_features_vec_np[:,-n:,0])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,0])
        bsum_left = np.sum(train_features_vec_np[:,:n,0])

        #max_dict = {0:bsum_0,90:bsum_90,180:bsum_180,270:bsum_270}
        #predict_label =max(max_dict.iteritems(), key=operator.itemgetter(1))[0]
        if(bsum_right == min(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec

def h11(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,1])
        bsum_right = np.sum(train_features_vec_np[:,-n:,1])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,1])
        bsum_left = np.sum(train_features_vec_np[:,:n,1])

        #max_dict = {0:bsum_0,90:bsum_90,180:bsum_180,270:bsum_270}
        #predict_label =max(max_dict.iteritems(), key=operator.itemgetter(1))[0]
        if(bsum_right == min(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec

def h12(train_data_np,N):
    predict_label_vec = []
    n = 3
    for i in range(0,N):
        train_features_vec_np = train_data_np[i]
        bsum_top = np.sum(train_features_vec_np[:n,:,2])
        bsum_right = np.sum(train_features_vec_np[:,-n:,2])
        bsum_bottom = np.sum(train_features_vec_np[-n:,:,2])
        bsum_left = np.sum(train_features_vec_np[:,:n,2])

        #max_dict = {0:bsum_0,90:bsum_90,180:bsum_180,270:bsum_270}
        #predict_label =max(max_dict.iteritems(), key=operator.itemgetter(1))[0]
        if(bsum_left == max(bsum_top,bsum_right,bsum_bottom,bsum_left)):
            predict_label_vec.append(1)
        else:
            predict_label_vec.append(0)
    return predict_label_vec


def read_params_from_modelfile():
    with open('adaboost_file.txt','r') as f:
        param_str = f.read()
    return param_str.split(" ")

def predict():
    count = 0
    param_str = read_params_from_modelfile()
    param = [float(i) for i in param_str]
    test_data= []
    test_label = []
    image_ids = []
    output_str = ''

    with open("test-data.txt") as f:
        for line in f.readlines():
            pixels = line.split(" ")
            image_ids.append(pixels[0])
            elements = pixels[2:]
            #print(int(pixels[1]))
            test_label.append(int(pixels[1]))
            new_data = []
            for i in range(0,len(elements),3):
                new_data.append(elements[i:i+3])
            X = []
            for i in range(0,len(new_data),8):
                X.append(new_data[i:i+8])
            test_data.append(X)
    test_data_np = np.array(test_data,dtype=int)
    test_label_np = np.array(test_label,dtype = int)
    #print(test_data_np.shape)
    #print(test_label_np)
    for index in range(0,test_data_np.shape[0]):
        #print(test_data_np[i].shape)
        my_predict  = {}
        predicted_label1 = h1([test_data_np[index]],1)
        predicted_label2 = h2([test_data_np[index]],1)
        predicted_label3 = h3([test_data_np[index]],1)
        my_predict[1] = predicted_label1[0]
        my_predict[2] = predicted_label2[0]
        my_predict[3] = predicted_label3[0]


        predicted_label4 = h4([test_data_np[index]],1)
        predicted_label5 = h5([test_data_np[index]],1)
        predicted_label6 = h6([test_data_np[index]],1)
        my_predict[4] = predicted_label4[0]
        my_predict[5] = predicted_label5[0]
        my_predict[6] = predicted_label6[0]


        predicted_label7 = h7([test_data_np[index]],1)
        predicted_label8 = h8([test_data_np[index]],1)
        predicted_label9 = h9([test_data_np[index]],1)

        my_predict[7] = predicted_label7[0]
        my_predict[8] = predicted_label8[0]
        my_predict[9] = predicted_label9[0]

        predicted_label10 = h10([test_data_np[index]],1)
        predicted_label11 = h11([test_data_np[index]],1)
        predicted_label12 = h12([test_data_np[index]],1)

        my_predict[10] = predicted_label10[0]
        my_predict[11] = predicted_label11[0]
        my_predict[12] = predicted_label12[0]
        sorted_x = sorted(my_predict.items(),key=operator.itemgetter(0))
        mylabel = {}
        for pair in sorted_x:
            if(pair[1] == 1):
                if(pair[0]>=1 and pair[0]<=3):
                    if(0 in mylabel.keys()):
                        mylabel[0]+=param[pair[0]-1]
                    else:
                        mylabel[0] = param[pair[0]-1]
                elif(pair[0]>=4 and pair[0]<=6):
                     if(90 in mylabel.keys()):
                         mylabel[90]+=param[pair[0]-1]
                     else:
                         mylabel[90] = param[pair[0]-1]
                elif(pair[0]>=7 and pair[0]<=9):
                    if(180 in mylabel.keys()):
                        mylabel[180]+=param[pair[0]-1]
                    else:
                        mylabel[180] = param[pair[0]-1]
                else:
                    if(270 in mylabel.keys()):
                        mylabel[270]+=param[pair[0]-1]
                    else:
                        mylabel[270] = param[pair[0]-1]
        if(len(mylabel) == 0):
            print("I dont know what to do")
            predict = 0
        else:
            predict = max(mylabel.iteritems(), key=operator.itemgetter(1))[0]
        if(predict == test_label_np[index]):
            count+=1
        else:
            #print("Predict wrong %d,%d,%s"%(predict,test_label_np[index],image_ids[index]))
            pass
        output_str +=image_ids[index] + ' ' + str(predict) + '\n'

    print("Accuracy : %f"%((count/test_data_np.shape[0]) * 100))
    with open('output.txt','w') as f:
        f.write(output_str)


def train():
    train_label_vec = []
    global N
    global W
    global Z
    train_features_vec = []
    count = 0
    examples = 0
    test_string = "train-data.txt"
    n = 4
    train_label = []
    train_data = []
    with open(test_string) as f:
        for line in f.readlines():
            examples+=1
            pixels = line.split(" ")
            elements = pixels[2:]
            train_label.append(int(pixels[1]))
            new_data = []
            for i in range(0,len(elements),3):
                new_data.append(elements[i:i+3])
            X = []
            for i in range(0,len(new_data),8):
                X.append(new_data[i:i+8])
            train_data.append(X)


    train_data_np = np.array(train_data,dtype=int)
    train_label_np = np.array(train_label,dtype = int)
    N = len(train_data)
    H = {h1:0,h2:0,h3:0,h4:90,h5:90,h6:90,h7:180,h8:180,h9:180,h10:270,h11:270,h12:270}


    AdaBoost(train_data_np,train_label_np,H)
    print("*************training done***********************")
    temp = {}
    for k,v in Z.iteritems():
        if(k == h1):
            temp[1] = v
        elif(k == h2):
            temp[2] = v
        elif(k == h3):
            temp[3] = v
        elif(k == h4):
            temp[4] = v
        elif(k == h5):
            temp[5] = v
        elif(k == h6):
            temp[6] = v
        elif(k == h7):
            temp[7] = v
        elif(k == h8):
            temp[8] = v
        elif(k == h9):
            temp[9] = v
        elif(k == h10):
            temp[10] = v
        elif(k == h11):
            temp[11] = v
        elif(k == h12):
            temp[12] = v
    temp1 = sorted(temp.items(),key=operator.itemgetter(0))
    param_str = ''
    for i in temp1:
        param_str += str(i[1]) + ' '
    param_str = param_str.strip()
    with open('adaboost_file.txt','w') as f:
        f.write(param_str)

train()
predict()
