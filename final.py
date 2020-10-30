#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from sklearn.externals import joblib
# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

import cv2
import numpy as np 
import os
from datetime import date 

import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input,Convolution2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.utils import np_utils
import tensorflow as tf
from keras.models import load_model

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model



# In[2]:


def showtable(date):
    nos=[]
    file_name=date+".txt"
    with open(file_name,"a+") as f:
        f.seek(0,0)
        nos=f.readlines()
    nos=[a[:len(a)-1] for a in nos]

    dic={}
    data=[]
    with open("data.txt","a+") as f:
        f.seek(0,0)
        data=f.readlines()
    data=[a[:len(a)-1] for a in data]
    for temp in data:
        ls=temp.split(" ")
        dic[ls[0]]=ls[1]+" "+ ls[2]
    space=13
    print("Roll NO."," "*space,"|""NAME"," "*space,"|""DEPARTMENT"," "*space,"|","DATE")
    print()
    for temp in nos:
        namedept=dic[temp].split(" ")
        print(temp," "*(8-len(temp)+space),"|",namedept[0]," "*(4-len(namedept[0])+space),"|",namedept[1]," "*(10-len(namedept[1])+space),"|",date[0:4]+"/"+date[4:6]+"/"+date[6:]);
    
    


# In[3]:


def test():
    
    #Init Camera
    cap = cv2.VideoCapture(0)

    # Face Detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    skip = 0
    dataset_path = 'data/'



    class_id = 0 # Labels for the given file
    names = {} #Mapping btw id - name


    # Data Preparation for labeling
    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            #Create a mapping btw class_id and name
            #class_id =int(fx[:-4]) 
            names[class_id] = fx[:-4]
            print("Loaded "+fx)
            data_item = np.load(dataset_path+fx)

            class_id=class_id+1
            #Create Labels for the class
            target = class_id*np.ones((data_item.shape[0],))
            




    #joblib_file = "joblib_model.pkl"


    # Load from file
    #classifier = joblib.load(joblib_file)
    model = load_model("model.h5")
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

    # Testing 

    while True:
        ret,frame = cap.read()
        if ret == False:
            continue

        faces = face_cascade.detectMultiScale(frame,1.3,5)
        if(len(faces)==0):
            continue

        for face in faces:
            x,y,w,h = face

            #Get the face ROI
            offset = 17
            face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
            
            face_section = cv2.resize(face_section,(28,28))
            image = tf.cast(face_section, tf.float32)
            #Predicted Label (out)
            #out = knn(trainset,face_section.flatten())
            img = np.reshape(image,[1,28,28,3])
            out = model.predict_classes(img)
            #print(out)

            #Display on the screen the roll no. and rectangle around it
            pred_name = names[out[0]]
            
            ##from here you have to call sql database function and pass pred_name as argument  ----- 
            file_name_date=date.today()
            file_name=file_name_date.strftime('%Y%m%d')+".txt"
            flag=1
            nos=[]
            with open(file_name,"a+") as f:
                f.seek(0,0)
                nos=f.readlines()
            nos=[a[:len(a)-1] for a in nos]
            
            if pred_name not in nos:
                with open(file_name,"a+") as f:
                    f.write(pred_name+'\n');

            cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        cv2.imshow("Faces",frame)

        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()







# In[4]:


def train():
    #Init Camera
    cap = cv2.VideoCapture(0)

    # Face Detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    skip = 0
    face_data = []
    dataset_path = 'data/'
    file_name = input("Enter the roll number of the person : ")
    person_name=input("Enter the name of person :  ")
    dep_name=input("Enter Department name : ")
    final=file_name+" "+person_name+" "+dep_name+"\n"
    with open("data.txt","a+") as f:
        f.write(final);
    
    
    while True:
        ret,frame = cap.read()

        if ret==False:
            continue

        #gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(frame,1.3,5)
        if len(faces)==0 :
            continue

        faces = sorted(faces,key=lambda f:f[2]*f[3])

        # Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
        for face in faces[-1:]:
            x,y,w,h = face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

            #Extract (Crop out the required face) : Region of Interest
            offset = 17
            face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_section = cv2.resize(face_section,(28,28))

            skip += 1
            if skip%10==0:
                face_data.append(face_section)
                print(len(face_data))


        cv2.imshow("Frame",frame)
        #cv2.imshow("Face Section",face_section)

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break

    # Convert our face list array into a numpy array
    face_data = np.asarray(face_data)
    print(face_data.shape)
    face_data = face_data.reshape((face_data.shape[0],-1))
    print(face_data.shape)

    # Save this data into file system
    np.save(dataset_path+file_name+'.npy',face_data)


    ##Training
    face_data = [] 
    labels = []

    class_id = 0 # Labels for the given file
    names = {} #Mapping btw id - name

    # Data Preparation
    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            #Create a mapping btw class_id and name
            #class_id =int(fx[:-4])
            names[class_id] = fx[:-4]
            print("Loaded "+fx)
            data_item = np.load(dataset_path+fx)
            face_data.append(data_item)

            #Create Labels for the class
            target = class_id*np.ones((data_item.shape[0],))
            class_id += 1
            labels.append(target)

    face_dataset = np.concatenate(face_data,axis=0)
    face_labels = np.concatenate(labels,axis=0)

    print(face_dataset.shape)
    print(face_labels.shape)

    #classifier = svm.SVC(gamma=0.001)
    #fit to the trainin data
    #classifier.fit(face_dataset,face_labels)
    #print(classifier.score(face_dataset,face_labels))
    
    X_train = face_dataset.reshape((-1,28,28,3))
    Y_train = np_utils.to_categorical(face_labels)
    classes=Y_train.shape[1];
    print(classes)
    
    
    
    model = Sequential()
    model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,3)))
    model.add(Convolution2D(64,(3,3),activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(2,2))
    model.add(Convolution2D(32,(5,5),activation='relu'))
    model.add(Convolution2D(8,(5,5),activation='relu'))
    model.add(Flatten())
    model.add(Dense(classes,activation='softmax'))
    model.summary()
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    model.fit(X_train,Y_train,epochs=20,shuffle=True,batch_size=4,validation_split=0.20)
    
    
  
    


    # Save to the model in the current working directory
    #joblib_file = "joblib_model.pkl"
    #joblib.dump(model, joblib_file)
    model.save("model.h5")
    print("Data Successfully saved and Trained")

    cap.release()
    cv2.destroyAllWindows()


# In[5]:


if __name__=="__main__":
    
    while 1:
        print()
        print()
        print("Press 1 for new entry/to update entry")
        print("Press 2 to run for attendance")
        print("Press 3 to see the attendace history")
        print("press 4 to Quite")

        
        x=int(input("Enter Your Choice:  "))
        if x==1:
            train()
        elif x==2:
            test()
        elif x==3:
            date=input("Enter Date in YYYYMMDD Format")
            print("\n\n\n")
            showtable(date)
            print("\n\n\n")
        else:
            break
    
    
   


# In[ ]:




