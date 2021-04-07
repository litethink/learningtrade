import pandas as pd
import numpy as np

data = pd.read_csv("data.txt",header=None,names=["怀孕次数","口服葡萄糖耐量试验中血浆葡萄糖浓度","舒张压（mm Hg）","三头肌组织褶厚度（mm）","2小时血清胰岛素（μU/ ml）","体重指数（kg/（身高(m)）^ 2）","糖尿病系统功能","年龄（岁）","是否患有糖尿病"])

#把属性值为0的地方转换成NaN
data.iloc[:,0:8]=data.iloc[:,0:8].applymap(lambda x:np.NaN if x==0 else x)
data = data.dropna(how="any",axis=0)

#随机选取80%的样本作为训练样本
data_train = data.sample(frac=0.8,random_state=4,axis=0)

#剩下的作为测试样本
testset_idx = [i for i in data.index.values if i not in data_train.index.values]
data_test = data.loc[testset_idx,:]
X_testset = data_test.iloc[:,:-1]
y_testset = data_test.iloc[:,-1]

#提取X特征和y目标
#训练集
X_train = data_train.iloc[:,:-1]
y_train = data_train.iloc[:,-1]

X, y = np.asarray(X_train, dtype="float32"), np.asarray(y_train, dtype="float32")


'''fit(X,y)'''

y = y.reshape(X.shape[0],1)
#总样本数
samples_len = len(y)

data = np.hstack((X,y))


#按类别分隔数据
#类标签
labels = np.unique(y)
#类标签总数
labels_len = len(labels)

data_byclass = {} 
for i in labels:
    data_byclass[i] = data[data[:,-1]==i]

def cal_prior_prob(y_byclass):
    ###计算y的先验概率（使用拉普拉斯平滑）###
    ###输入当前类别下的目标，输出该目标的先验概率###
    #计算公式：（当前类别下的样本数+1）/（总样本数+类标签总数）
    return (len(y_byclass) + 1) / (samples_len + labels_len)   



#fit 返回的结果 先验概率，特征的平均值,方差:  
prior_prob = []
X_vars  = []
X_means = []
 
for i in labels:
    X_byclass = data_byclass[i][:,:-1]
    y_byclass = data_byclass[i][:,-1]
    prior_prob.append(cal_prior_prob(y_byclass))
    X_vars.append(X_byclass.var(axis=0))
    X_means.append(X_byclass.mean(axis=0))
    



'''predict(X_test)'''

X_test = X_testset.values[0]

def cal_gaussian_prob(X_test, mean, var):
    ###计算训练集特征（符合正态分布）在各类别下的条件概率###
    ###输入新样本的特征，训练集特征的平均值和方差，输出新样本的特征在相应训练集中的分布概率###
    #计算公式：(np.exp(-(X_new-mean)**2/(2*var)))*(1/np.sqrt(2*np.pi*var))
    gaussian_prob = []
    for a,b,c in zip(X_test, mean, var):
        formula1 = np.exp(- np.square(a-b) / (2*c))
        formula2 = 1 / np.sqrt(2*np.pi*c)
        gaussian_prob.append(formula2*formula1)
    return gaussian_prob

posteriori_prob = []
for prob,mean,var in zip(prior_prob, X_means, X_vars):
    gaussian = cal_gaussian_prob(X_test,mean,var)
    # posteriori_prob <=log(prob) + sum(log([f1(a1,b1,c1)*f2(c1),f1(a2,b2,c1)*f2(c2),...]))
    posteriori_prob.append(np.log(prob) + sum(np.log(gaussian)))
idx = np.argmax(posteriori_prob)

result = labels[idx]
