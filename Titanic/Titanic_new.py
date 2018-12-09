#-*-coding:utf-8-*-

import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
print(data.columns)
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Cabin','Embarked']]

data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Cabin']=pd.factorize(data.Cabin)[0]
data.fillna(0, inplace=True)
data['Sex']=[1 if x=='male' else 0 for x in data.Sex]

data['p1']=np.array(data['Pclass']==1).astype(np.int32)
data['p2']=np.array(data['Pclass']==2).astype(np.int32)
data['p3']=np.array(data['Pclass']==3).astype(np.int32)

del data['Pclass']

data.Embarked.unique()
data['e1']=np.array(data['Embarked']=='S').astype(np.int32)
data['e2']=np.array(data['Embarked']=='C').astype(np.int32)
data['e3']=np.array(data['Embarked']=='Q').astype(np.int32)

del data['Embarked']
data.values.dtype
data_train = data[['Sex',  'Age', 'SibSp', 'Parch', 'Fare','Cabin','p1','p2','p3','e1','e2','e3']]
data_target = data['Survived'].values.reshape(len(data),1)

import tensorflow as tf
x=tf.placeholder("float",shape=[None,12])
y=tf.placeholder("float",shape=[None,1])

weight = tf.Variable(tf.random_normal([12,1]))
bias =tf.Variable(tf.random_normal([1]))
output =tf.matmul(x,weight)+bias
pred =tf.cast(tf.digmoid(output)>0.5,tf.float32)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=output))
train_step=tf.train.GradientDescentOptimizer(0.0003).minimize(loss)
accuracy=tf.reduce_mean(tf.cast(tf.equal(pred,y),tf.float32))

data_test = pd.read_csv('test.csv')
data_test = data_test[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Cabin','Embarked']]

data_test['Age'] = data_test['Age'].fillna(data['Age'].mean())
data_test['Cabin']=pd.factorize(data_test.Cabin)[0]
data_test.fillna(0, inplace=True)
data_test['Sex']=[1 if x=='male' else 0 for x in data.Sex]

data_test['p1']=np.array(data['Pclass']==1).astype(np.int32)
data_test['p2']=np.array(data['Pclass']==2).astype(np.int32)
data_test['p3']=np.array(data['Pclass']==3).astype(np.int32)

del data['Pclass']

data.Embarked.unique()
data['e1']=np.array(data['Embarked']=='S').astype(np.int32)
data['e2']=np.array(data['Embarked']=='C').astype(np.int32)
data['e3']=np.array(data['Embarked']=='Q').astype(np.int32)

del data['Embarked']