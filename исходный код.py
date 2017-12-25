import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
X = [[1,2], [1,4], [2,1], [2,3], [3,2], [3,4], [4,1], # точки красного класса соотвутствует значение 0 массива y[]
     [4.7, 3.7], [5.1, 3.1], [5, 3.5], [5.5, 3], [5.5, 3.4], [4.5, 3.3]] # точки синего класса соотвутствует значение 1 y
y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] # масив ответов, в котором указывается к какому классу принадлежат точки сверху.
# размеры массивов X[] и y[] должны быть равны так одной точке соответствует один ответ
u = [4, 3] # точка, которую необходимо классифицировать к одному из двух классов
u = np.array(u)
u_class = [-1] # класс к которому принадолежит точка пока не известен
X = np.array(X)

colors = ListedColormap(['red', 'blue']) #один цвет соответствует одному классу
plt.scatter(X[:,0], X[:,1], c=y, cmap=colors, s=50)
plt.title("fris-stolp") # надпись сверху над изображением


n = len(X)  # размер массива - количество точек
xl = np.arange(n)  # data indexes

eps = 1e-7 # так как делить на 0 нельзя, то к знаменателю лучше прибавить маленькое число
alpha = 0.5
tetta = 0.2



# Функция конкурентного сходства или FRiS-функция
def fris(a, b, x):
    return (distance.euclidean(a, x) - distance.euclidean(a, b)) / (distance.euclidean(a, x) + distance.euclidean(a, b) + eps)

# возвращает ближайший к u объект из U
def nearest_neighbor(u, U):
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(U)
    return U[nbrs.kneighbors(u, return_distance=False)]


class_val = [0, 0]
for i, val_i in enumerate(X):
    for j, val_j in enumerate(X):
        if (fris(val_i, val_j, u) < 0):
            class_val[0] = class_val[0] + 1
        else:
            class_val[1] = class_val[1] + 1

neigh = KNeighborsClassifier(n_neighbors=10)  # объявляю объект класса KNeighborsClassifier
neigh.fit(X, y) # помещаю в экземпляр класса данные
u = np.reshape(u, (1,2))

#------------------

classes_cnt = 1
prev = y[0]
for i, val in enumerate(y) :
    if (val != prev):
        prev = val
        classes_cnt = classes_cnt + 1

nearest_n = [list() for t in range(classes_cnt)]
for i in range(classes_cnt):
    X_class = []
    cnt = 0
    for j, val in enumerate(X):
        if (y[j] == i):
            X_class.append(val)
            cnt = cnt+1
    X_class = np.array(X_class)
    nearest_n[i] = nearest_neighbor(u, X_class)


if (fris(nearest_n[0], nearest_n[1], u) < 0):# если фрис функция принимает отрицательное значение, то классифицируем к красному классу
    plt.scatter(u[0][0], u[0][1], c=u_class, cmap=ListedColormap(['red']), s=100, marker = 's')
else:# если фрис функция принимает отрицательное значение, то классифицируем к синему классу
    plt.scatter(u[0][0], u[0][1], c=u_class, cmap=ListedColormap(['blue']), s=100, marker = 's')


