# 利用sklearn的乳腺癌数据集进行svm二分类任务

import pandas as pd
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics

# 加载数据集
breast_cancer=load_breast_cancer()
data=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
target=pd.DataFrame(breast_cancer.target,columns={"diagnosis"})
# 数据探索
print(data.head(5))
print(data.info())
print(target.head(5))
print(target.info())

# 将特征字段分成 3 组
features_mean= list(data.columns[0:10])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:30])
# 数据清洗


# 将肿瘤诊断结果可视化
sns.countplot(target['diagnosis'],label="Count")
plt.show()
# 用热力图呈现 features_mean 字段之间的相关性
corr = data[features_mean].corr()
plt.figure(figsize=(14,14))
# annot=True 显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.show()

# 特征选择
features_remain = ['mean radius','mean texture', 'mean smoothness','mean compactness','mean symmetry', 'mean fractal dimension'] 
new_data=data[features_remain]
# 抽取 30% 的数据作为测试集，其余作为训练集
train_X,test_X, train_y,test_y = train_test_split(new_data,target, test_size = 0.3)# in this our main data is splitted into train and test

# 采用 Z-Score 规范化数据，保证每个特征维度的数据均值为 0，方差为 1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

# 创建 SVM 分类器
model = svm.SVC()
# 用训练集做训练
model.fit(train_X,train_y)
# 用测试集做预测
prediction=model.predict(test_X)
print('准确率: ', metrics.accuracy_score(prediction,test_y))


