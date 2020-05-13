# -*- coding: utf-8 -*-
"""
Created on Mon May 11 01:34:34 2020

@author: chongyi
"""

import pickle

with open('serve_left_block_left_win.pickle','rb') as file:
    datalist1=pickle.load(file)
with open('serve_left_block_right_loss.pickle','rb') as file:
    datalist2=pickle.load(file)
with open('serve_right_block_left_loss.pickle','rb') as file:
    datalist3=pickle.load(file)
with open('serve_right_block_right_loss.pickle','rb') as file:
    datalist4=pickle.load(file)
with open('serve_right_block_right_win_2P.pickle','rb') as file:
    datalist5=pickle.load(file)
with open('serve_right_block_left_loss_2P.pickle','rb') as file:
    datalist6=pickle.load(file)
with open('serve_left_365frame_2P.pickle','rb') as file:
    datalist7=pickle.load(file)
with open('serve_left_661frame2P.pickle','rb') as file:
    datalist8=pickle.load(file)
with open('serve_left_block_right_new_1148frame.pickle','rb') as file:
    datalist9=pickle.load(file)
frame=[]
ball=[]
ball_speed=[]
blocker=[]
platform_1P=[]
platform_2P=[]
command_1P=[]
command_2P=[]

for j in range(0,len(datalist1)-1):
    i=j%len(datalist1)
    frame.append(datalist1[i]['frame'])
    ball.append(datalist1[i]['ball'])
    ball_speed.append(datalist1[i]['ball_speed'])
    blocker.append(datalist1[i]['blocker'])
    platform_1P.append(datalist1[i]['platform_1P'])
    platform_2P.append(datalist1[i]['platform_2P'])
    command_1P.append(datalist1[i]['command_1P'])
    command_2P.append(datalist1[i]['command_2P'])

for j in range(0,len(datalist2)-150):
    i=j%len(datalist2)
    frame.append(datalist2[i]['frame'])
    ball.append(datalist2[i]['ball'])
    ball_speed.append(datalist2[i]['ball_speed'])
    blocker.append(datalist2[i]['blocker'])
    platform_1P.append(datalist2[i]['platform_1P'])
    platform_2P.append(datalist2[i]['platform_2P'])
    command_1P.append(datalist2[i]['command_1P'])
    command_2P.append(datalist2[i]['command_2P'])

for j in range(0,len(datalist3)-100):
    i=j%len(datalist3)
    frame.append(datalist3[i]['frame'])
    ball.append(datalist3[i]['ball'])
    ball_speed.append(datalist3[i]['ball_speed'])
    blocker.append(datalist3[i]['blocker'])
    platform_1P.append(datalist3[i]['platform_1P'])
    platform_2P.append(datalist3[i]['platform_2P'])
    command_1P.append(datalist3[i]['command_1P'])
    command_2P.append(datalist3[i]['command_2P'])

for j in range(0,len(datalist4)-100):
    i=j%len(datalist4)
    frame.append(datalist4[i]['frame'])
    ball.append(datalist4[i]['ball'])
    ball_speed.append(datalist4[i]['ball_speed'])
    blocker.append(datalist4[i]['blocker'])
    platform_1P.append(datalist4[i]['platform_1P'])
    platform_2P.append(datalist4[i]['platform_2P'])
    command_1P.append(datalist4[i]['command_1P'])
    command_2P.append(datalist4[i]['command_2P'])

for j in range(0,len(datalist5)-1):
    i=j%len(datalist5)
    frame.append(datalist5[i]['frame'])
    ball.append(datalist5[i]['ball'])
    ball_speed.append(datalist5[i]['ball_speed'])
    blocker.append(datalist5[i]['blocker'])
    platform_1P.append(datalist5[i]['platform_1P'])
    platform_2P.append(datalist5[i]['platform_2P'])
    command_1P.append(datalist5[i]['command_1P'])
    command_2P.append(datalist5[i]['command_2P'])

for j in range(0,len(datalist6)-100):
    i=j%len(datalist6)
    frame.append(datalist6[i]['frame'])
    ball.append(datalist6[i]['ball'])
    ball_speed.append(datalist6[i]['ball_speed'])
    blocker.append(datalist6[i]['blocker'])
    platform_1P.append(datalist6[i]['platform_1P'])
    platform_2P.append(datalist6[i]['platform_2P'])
    command_1P.append(datalist6[i]['command_1P'])
    command_2P.append(datalist6[i]['command_2P'])

for j in range(0,len(datalist7)-100):
    i=j%len(datalist7)
    frame.append(datalist7[i]['frame'])
    ball.append(datalist7[i]['ball'])
    ball_speed.append(datalist7[i]['ball_speed'])
    blocker.append(datalist7[i]['blocker'])
    platform_1P.append(datalist7[i]['platform_1P'])
    platform_2P.append(datalist5[i]['platform_2P'])
    command_1P.append(datalist5[i]['command_1P'])
    command_2P.append(datalist5[i]['command_2P'])

for j in range(0,len(datalist8)-100):
    i=j%len(datalist8)
    frame.append(datalist8[i]['frame'])
    ball.append(datalist8[i]['ball'])
    ball_speed.append(datalist8[i]['ball_speed'])
    blocker.append(datalist8[i]['blocker'])
    platform_1P.append(datalist8[i]['platform_1P'])
    platform_2P.append(datalist8[i]['platform_2P'])
    command_1P.append(datalist8[i]['command_1P'])
    command_2P.append(datalist8[i]['command_2P'])

for j in range(0,len(datalist9)-1):
    i=j%len(datalist9)
    frame.append(datalist9[i]['frame'])
    ball.append(datalist9[i]['ball'])
    ball_speed.append(datalist9[i]['ball_speed'])
    blocker.append(datalist9[i]['blocker'])
    platform_1P.append(datalist9[i]['platform_1P'])
    platform_2P.append(datalist9[i]['platform_2P'])
    command_1P.append(datalist9[i]['command_1P'])
    command_2P.append(datalist9[i]['command_2P'])


#preprocessing    
import numpy as np

#preprocess frame
Frame=np.array(frame)
#preprocess platform
PlatX_1P=np.array(platform_1P)[:,0][:, np.newaxis]
PlatX_2P=np.array(platform_2P)[:,0][:, np.newaxis]

#preprocess ball
BallX=np.array(ball)[:,0][:,np.newaxis]
BallY=np.array(ball)[:,1][:,np.newaxis]
Ball_SpeedX=np.array(ball_speed)[:,0][:,np.newaxis]
Ball_SpeedY=np.array(ball_speed)[:,1][:,np.newaxis]

#preprocess blocker
BlockerX=np.array(blocker)[:,0][:,np.newaxis]
BlockerY=np.array(blocker)[:,1][:,np.newaxis]

#preprocess commend
Command_1P=np.array(command_1P)[:][:,np.newaxis]
Command_2P=np.array(command_2P)[:][:,np.newaxis]
for i in range(len(Command_1P)):
    if(Command_1P[i]=='NONE'):
        Command_1P[i]=0
        Command_2P[i]=0
    elif(Command_1P[i]=='MOVE_RIGHT'):
        Command_1P[i]=1
        Command_2P[i]=1
    elif(Command_1P[i]=='MOVE_LEFT'):
        Command_1P[i]=2
        Command_2P[i]=2
    elif(Command_1P[i]=='SERVE_TO_LEFT'):
        Command_1P[i]=3
        Command_2P[i]=3
    elif(Command_1P[i]=='SERVE_TO_RIGHT'):
        Command_1P[i]=4
        Command_2P[i]=4

#transform to csv
import pandas as pd
data=[[],[],[],[],[],[],[],[],[],[],[]]
for i in range(0,len(PlatX_1P)):
    data[0].append(PlatX_1P[i])
for i in range(0,len(PlatX_2P)):
    data[1].append(PlatX_2P[i])
for i in range(0,len(BallX)):
    data[2].append(BallX[i])
for i in range(0,len(BallY)):
    data[3].append(BallY[i])
for i in range(0,len(Ball_SpeedX)):
    data[4].append(Ball_SpeedX[i])
for i in range(0,len(Ball_SpeedY)):
    data[5].append(Ball_SpeedY[i])
for i in range(0,len(BlockerX)):
    data[6].append(BlockerX[i])
for i in range(0,len(BlockerY)):
    data[7].append(BlockerY[i])
for i in range(0,len(Command_1P)):
    data[8].append(Command_1P[i])
for i in range(0,len(Command_2P)):
    data[9].append(Command_2P[i])
for i in range(0,len(Frame)):
    data[10].append(Frame[i])
data=(list(map(list, zip(*data))))
csv=pd.DataFrame(data=data)
csv.to_csv('testcsv.csv',encoding='gbk')

#preprocessing
dataset=csv
X_1P=dataset.iloc[:,[0,2,3,4,5,6,7,10]].values
Y_1P=dataset.iloc[:,8].values
Y_1P=np.asarray(Y_1P,'int32')
X_2P=dataset.iloc[:,[1,2,3,4,5,6,7,10]].values
Y_2P=dataset.iloc[:,9].values
Y_2P=np.asarray(Y_2P,'int32')

#spilt
from sklearn.model_selection import train_test_split
X_1P_train,X_1P_test,y_1P_train,y_1P_test=train_test_split(X_1P,Y_1P,test_size=0.1,random_state=10)
X_2P_train,X_2P_test,y_2P_train,y_2P_test=train_test_split(X_2P,Y_2P,test_size=0.1,random_state=10)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)
Y_Train = sc.fit_transform(Y_Train)
Y_Test = sc.transform(Y_Test)
"""
"""
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train,y_train)
"""
from sklearn.ensemble import RandomForestClassifier
classifier_1P=RandomForestClassifier(random_state=0)
classifier_1P.fit(X_1P_train,y_1P_train)
classifier_2P=RandomForestClassifier(random_state=0)
classifier_2P.fit(X_2P_train,y_2P_train)


# Predicting the Test set results
y_pred_1P = classifier_1P.predict(X_1P_test)
y_pred_2P = classifier_2P.predict(X_2P_test)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_1P_test, y_pred_1P)
cm2 = confusion_matrix(y_2P_test, y_pred_2P)

#save model
filename="rfc_1P.sav"
pickle.dump(classifier_1P,open(filename, 'wb'))
filename="rfc_2P.sav"
pickle.dump(classifier_2P,open(filename, 'wb'))




