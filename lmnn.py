import numpy as np
from common import load_mnist
from knn import KNN_predict
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric  

from metric_learn import LMNN

def knn(train_x, train_y, test_x, test_y):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_x, train_y)
    acc = (neigh.predict(test_x) == test_y).sum()
    return float(acc))/test_y.shape[0]

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = load_mnist(percentage=0.01, skip_valid=True)

    pca = PCA(whiten=True)
    pca.fit(train_x)
    components, variance = 0, 0.0
    for components, ele in enumerate(pca.explained_variance_ratio_):
        variance += ele
        if variance > 0.90: break
    components += 1
    print 'n_components=%d'%components
    pca.set_params(n_components=components)
    pca.fit(train_x)

    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)

    lmnn = LMNN(k=5, learn_rate=1e-5, max_iter=200) 
    L = lmnn.fit(train_x, train_y, verbose=True).L

    knn(train_x, train_y, test_x, test_y)
    knn(np.dot(train_x, L), train_y, np.dot(test_x, L), test_y)


