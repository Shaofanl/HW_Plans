from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA, SparsePCA
import numpy as np

def load_data(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('@'): continue
            l = map(float, line.strip().split(','))

            feature = l[:-1]
            x.append(feature)
            y.append(1)

            # error example
            for _ in xrange(100):
                r = np.random.randint(0, len(feature))
                feature[r] = 1-feature[r]
            x.append(feature)
            y.append(0)

    return np.array(x), np.array(y)



if __name__ == '__main__':
    x, y = load_data('./parsed.arff')
    print x.shape, y.shape

    pca = PCA()
    pca.fit(x)
    acc = 0
    for ind, ele in enumerate(pca.explained_variance_ratio_):
        acc += ele
        if acc > 0.010: break;
    print 'Decomposition to {} dim'.format(ind+1)
    pca = PCA(n_components=ind+1)
    x = pca.fit_transform(x)

    for _ in xrange(10):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


        clf = svm.SVC(kernel='rbf')
        clf.fit(X_train, y_train)

        correct = (clf.predict(X_test) == y_test).sum()
        print "{}/{} = {}".format(correct, y_test.shape[0], float(correct)/y_test.shape[0])

