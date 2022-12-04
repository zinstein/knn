import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

stop_list = pd.read_csv("cn_stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')

def txt_cut(juzi):
    lis = [w for w in jieba.lcut(juzi) if w not in stop_list.values]
    lis = [x.strip() for x in lis if x.strip() != '']
    return " ".join(lis)


df = pd.read_excel('gastric.xlsx')
data = pd.DataFrame()
data['label'] = df['Label']
data['word'] = df['Text'].astype('str').apply(txt_cut)

# 将文件分割成单字, 建立词索引字典
tok = Tokenizer(num_words=6000)
tok.fit_on_texts(data['word'].values)
print("样本数 : ", tok.document_count)
print({k: tok.word_index[k] for k in list(tok.word_index)[:20]})


X = tok.texts_to_sequences(data['word'].values)
# #查看x的长度的分布
# length=[]
# for i in X:
#     length.append(len(i))
# v_c=pd.Series(length).value_counts()
# print(v_c[v_c>1])  #频率大于50才展现
# v_c[v_c>1].plot(kind='bar',figsize=(12,5))
# plt.show()

# 将序列数据填充成相同长度
X = pad_sequences(X, maxlen=120)
Y = data['label'].values
X = np.array(X)
Y = np.array(Y)

# 划分测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))


# 第三步 基于KNN的图像分类处理
# 求最佳k和p值
def searchBestPar():
    bestScore = 0
    bestK = 0
    bestP = 0
    # for k in range(1, 10):
    #     clf = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(X_train, Y_train)
    #     scor = clf.score(X_test,Y_test)
    #     print(str(k) + ":" + str(scor))
    #     if scor > bestScore:
    #         bestScore = scor
    #         bestK = k
    for k in range(1, 20):
        for p in range(1, 10):
            clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p).fit(X_train, Y_train)
            scor = clf.score(X_test, Y_test)
            if scor > bestScore:
                bestScore = scor
                bestK = k
                bestP = p
    print("best：" + str(bestK) + " " + str(bestP) + ":" + str(bestScore))


#searchBestPar()

clf = KNeighborsClassifier(n_neighbors=1, weights="distance", p=1).fit(X_train, Y_train)
predictions_labels = clf.predict(X_test)

print('算法评价:')
print((classification_report(Y_test, predictions_labels)))

# k = 0
# while k < 100:
#     print(X_test[k])
#     print(predictions_labels[k])
#     k = k + 1


# precision ( 精确度)：正确预测为正的，占全部预测为正的比例。
# recall（召回率）：正确预测为正的，占全部实际为正的比例。
# f1-score (f1值)：精确率和召回率的调和平均数。
# support （各分类样本的数量或测试集样本的总数量）。
# macro avg (宏平均值)：所有标签结果的平均值。
# weighted avg（加权平均值）：所有标签结果的加权平均值。
# accuracy是准确率，表示判定正确的次数与所有判定次数的比例
