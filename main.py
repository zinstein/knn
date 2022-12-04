import os
import re
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


# 第一步 切分训练集和测试集

X = []  # 定义图像名称
Y = []  # 定义图像分类类标
Z = []  # 定义图像像素

aa = 0
file_list = os.listdir("coil-20-proc//")
file_list.sort(key=lambda x: int(str(re.findall("\d+", x)[0])))
for f in file_list:
    # 获取图像名称
    X.append("coil-20-proc//" + str(f))
    Y.append(aa // 72 + 1)
    aa += 1

X = np.array(X)
Y = np.array(Y)

# 随机率为100% 选取其中的20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(len(X_train), len(X_test), len(y_train), len(y_test))

# 第二步 图像读取及转换为像素直方图

# 训练集
XX_train = []
for i in X_train:
    # 读取图像
    # print i
    image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 图像像素大小一致
    img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

    # 计算图像直方图并存储至X数组
    hist = cv2.calcHist([img], [0, 1], None,
                        [256, 256], [0.0, 255.0, 0.0, 255.0])

    XX_train.append(((hist / 255).flatten()))

# 测试集
XX_test = []
for i in X_test:
    # 读取图像
    # print i
    image = cv2.imdecode(np.fromfile(i, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 图像像素大小一致
    img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

    # 计算图像直方图并存储至X数组
    hist = cv2.calcHist([img], [0, 1], None,
                        [256, 256], [0.0, 255.0, 0.0, 255.0])

    XX_test.append(((hist / 255).flatten()))

#pca处理
pca = PCA(n_components = 65, whiten=True).fit(XX_train,y_train)
XX_train_pca = pca.transform(XX_train)
XX_test_pca = pca.transform(XX_test)
#lda处理
lda = LinearDiscriminantAnalysis(n_components =14).fit(XX_train_pca,y_train)
XX_train_lda = lda.transform(XX_train_pca)
XX_test_lda = lda.transform(XX_test_pca)

# 第三步 基于KNN的图像分类处理
#求最佳k和p值
def searchBestPar():
    bestScore=0
    bestK=1
    bestP=1
    # for k in range(1, 10):
    #     clf = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(XX_train, y_train)
    #     scor = clf.score(XX_test,y_test)
    #     if scor > bestScore:
    #         bestScore = scor
    #         bestK = k
    for k in range(1,20):
        for p in range(1, 10):
            clf = KNeighborsClassifier(n_neighbors=k,weights="distance",p=p).fit(XX_train_lda, y_train)
            scor=clf.score(XX_test_lda,y_test)
            print(str(k)+" "+str(p)+":"+str(scor))
            if scor > bestScore:
                bestScore = scor
                bestK = k
                bestP = p
    print("best："+str(bestK)+" "+str(bestP)+":"+str(bestScore))

#searchBestPar()

clf = KNeighborsClassifier(n_neighbors=5,weights="distance",p=3).fit(XX_train_lda, y_train)
predictions_labels = clf.predict(XX_test_lda)

print('预测结果:')
print(predictions_labels)
print('算法评价:')
print((classification_report(y_test, predictions_labels)))

# 输出前100张图片及预测结果
k = 0
while k < 100:
    # 读取图像
    print(X_test[k])
    image = cv2.imread(X_test[k])
    print(predictions_labels[k])
    # 显示图像
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    k = k + 1


# precision ( 精确度)：正确预测为正的，占全部预测为正的比例。
# recall（召回率）：正确预测为正的，占全部实际为正的比例。
# f1-score (f1值)：精确率和召回率的调和平均数。
# support （各分类样本的数量或测试集样本的总数量）。
# macro avg (宏平均值)：所有标签结果的平均值。
# weighted avg（加权平均值）：所有标签结果的加权平均值。
#accuracy是准确率，表示判定正确的次数与所有判定次数的比例