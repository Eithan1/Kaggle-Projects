#-*-coding:utf-8-*-

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer


#-------------------导入数据-------------------#
train_data = pd.read_csv('train.csv')




#-------------------观察分析数据-----------------------#
# print( train_data.describe() )
#
# sns.barplot(x='Sex', y='Survived', data=train_data)
# plt.show()
#
# sns.barplot(x='Embarked',y='Survived',hue='Sex',data=train_data)
# plt.show()
#
# sns.pointplot(x='Pclass',y='Survived',hue='Sex',data=train_data,
#              palette={'male':'blue','female':'pink'},
#              markers=['*','o'],linestyles=['-','--'])
# plt.show()
#
# grid = sns.FacetGrid(train_data, col='Survived', row='Sex', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()
#
# print( train_data.Sex.value_counts() )

# sns.barplot(x='SibSp',y='Survived',data=train_data)
# plt.show()
#
# sns.barplot(x='Parch',y='Survived',data=train_data)
# plt.show()

print('-------------------------------')
print ( train_data.Embarked.describe() )


#---------------------数据预处理--------------------------#
#年龄空白值用中位数填充
train_data.Age = train_data.Age.fillna(train_data.Age.median())
train_data.describe()

#将性别处理为数字，男性1，女性0
train_data['Sex'] = train_data['Sex'].apply(lambda s: 1 if s == 'male' else 0)

#用最多登船地S去填充登船地的空白，并将其转化为数字
train_data.Embarked = train_data.Embarked.fillna('S')
train_data.loc[train_data.Embarked == 'S','Embarked'] = 0
train_data.loc[train_data.Embarked == 'C','Embarked'] = 1
train_data.loc[train_data.Embarked == 'Q','Embarked'] = 2



#-------------------用逻辑回归训练模型-------------------#
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] #选取有用特征

alg = LogisticRegression()
kf = KFold(n_splits=5, random_state=1)  #k-折交叉验证
predictions = list()
for train, test in kf.split(train_data):
    k_train = train_data[features].iloc[train, :]
    k_label = train_data.Survived.iloc[train]
    alg.fit(k_train, k_label)
    k_predictions = alg.predict(train_data[features].iloc[test, :])
    predictions.append(k_predictions)

predictions = np.concatenate(predictions, axis=0)
print ( accuracy_score(train_data.Survived, predictions) )



#------------------------------处理需要预测数据------------------------#
test_data = pd.read_csv('test.csv')
test_data['Sex'] = test_data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
test_data.Embarked = test_data.Embarked.fillna('S')
test_data.loc[test_data.Embarked == 'S','Embarked'] = 0
test_data.loc[test_data.Embarked == 'C','Embarked'] = 1
test_data.loc[test_data.Embarked == 'Q','Embarked'] = 2


#--------------------------------开始预测--------------------------------#
test_data[features] = Imputer().fit_transform(test_data[features])
alg = LogisticRegression()
kf = KFold(n_splits=5,random_state=1)
for train, test in kf.split(train_data):
    k_train = train_data[features].iloc[train,:]
    k_label = train_data.Survived.iloc[train]
    alg.fit(k_train,k_label)
predictions = alg.predict(test_data[features])



#----------------------------------将预测结果写入文件-----------------------#
# df = DataFrame([test_data.PassengerId, Series(predictions)], index=['PassengerId', 'Survived'])
# df.T.to_csv('Titanic_submission.csv', index=False)
