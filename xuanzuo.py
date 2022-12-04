import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


stop_list = pd.read_csv("cn_stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')

def txt_cut(juzi):
    lis = [w for w in jieba.cut_for_search(juzi) if (w not in stop_list.values and len(w)!=1)]
    lis = [x.strip() for x in lis if x.strip() != '']
    return " ".join(lis)


df = pd.read_excel('gastric.xlsx')
data = pd.DataFrame()
data['label'] = df['Label']
data['word'] = df['Text'].astype('str').apply(txt_cut)

# 将文件分割成单字, 建立词索引字典
tok = Tokenizer(num_words=None)
tok.fit_on_texts(data['word'].values)
print("样本数 : ", tok.document_count)
print({k: tok.word_index[k] for k in list(tok.word_index)[:10]})
X = tok.texts_to_sequences(data['word'].values)
#查看x的长度的分布
# length=[]
# for i in X:
#     length.append(len(i))
# v_c=pd.Series(length).value_counts()
# print(v_c[v_c>1])
# v_c[v_c>1].plot(kind='bar',figsize=(12,5))
# plt.show()

# 将序列数据填充成相同长度
X = pad_sequences(X, maxlen=30)
Y = data['label'].values
X = np.array(X)
Y = np.array(Y)

# 划分测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))


# 超参数网格搜索
def searchBestPar():
    param_grid=[
        {
            'weights':['uniform'],
            'n_neighbors':[k for k in range(1,20)]
        },
        {
            'weights':['distance'],
            'n_neighbors':[k for k in range(1,20)],
            'p':[p for p in range(1,10)]
        }
    ]
    knn_clf=KNeighborsClassifier()
    grid_search=GridSearchCV(knn_clf,param_grid=param_grid,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,Y_train)
    print(grid_search.best_estimator_)
    print(grid_search.best_score_)

# searchBestPar()

clf = KNeighborsClassifier(n_neighbors=1, weights="distance", p=1).fit(X_train, Y_train)
predictions_labels = clf.predict(X_test)

print('算法评价:')
print((classification_report(Y_test, predictions_labels)))

labels = [1, 2, 3, 4, 5]
y_true = Y_test  # 正确标签
y_pred = predictions_labels  # 预测标签
tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(6, 4), dpi=120)
ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.savefig('matrix.png', format='png')
plt.show()

# precision ( 精确度)：正确预测为正的，占全部预测为正的比例。
# recall（召回率）：正确预测为正的，占全部实际为正的比例。
# f1-score (f1值)：精确率和召回率的调和平均数。
# support （各分类样本的数量或测试集样本的总数量）。
# accuracy是准确率，表示判定正确的次数与所有判定次数的比例
# macro avg (宏平均值)：所有标签结果的平均值。
# weighted avg（加权平均值）：所有标签结果的加权平均值。

