import os
import random
import cPickle
import numpy as np

from sklearn.decomposition import PCA
from sklearn.svm import SVC, NuSVC
from sklearn.cross_validation import train_test_split, KFold

from metric_learn import LMNN
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric  

def parse(filename, outputfilename, keywords, **kwargs):
#    parse('./data/blocks-lib', './parsed.arff', 
#            keywords=['(ON', '(STACK', '(UNSTACK', '(PICK-UP'],
#            RELATION = 'blocks-world')
 
    states_set = set([])
    with open(filename, 'r') as f:
        stage = None
        for line in f:
            if line.strip().startswith('(:'):
                stage = line.strip().split('(:')[-1]

            for keyword in keywords:
                if line.strip().startswith(keyword):
                    state_name = '{}:{}:{}'.format(stage, keyword, line.strip()).replace(' ', '-')
                    states_set.update([state_name])
    states_list = list(states_set)

    output = open(outputfilename, 'w')
    output.write('@RELATION {}\n'.format(kwargs.get('RELATION', 'UNKNOWN')))
    for state in states_set:
        output.write('@ATTRIBUTE {} {{1, 0}}\n'.format(state))
    output.write('@ATTRIBUTE SUCCESS {{1, 0}}\n'.format(state))
    output.write('@DATA\n'.format(state))

    features = []

    feature = []
    with open(filename, 'r') as f:
        stage = None
        for line in f:
            if line.strip().startswith('(:'):
                stage = line.strip().split('(:')[-1]
                if stage == 'trace':
                    if feature and max(feature) != 0:
                        # positive example
                        features.append(feature+[1])

                        # error example
                        for _ in xrange(3):
                            r = random.randint(0, len(states_set))
                            feature[r] = 1-feature[r]
                        features.append(feature+[0])

                    feature = [0]* len(states_set)

#            print line
            for keyword in keywords:
                if line.strip().startswith(keyword):
                    state_name = '{}:{}:{}'.format(stage, keyword, line.strip()).replace(' ', '-')

                    feature[states_list.index(state_name)] = 1 
                    print states_list.index(state_name)

    for feature in features:
        output.write(','.join(map(str, feature)))
        output.write(',1\n')

def change(filename, output, p):
    ifile = open(filename, 'r')
    ofile = open(output, 'w')
    for line in ifile:
        if any(map(lambda x: x in line, ['define', 'trace', 'init', 'plan', 'goal'])): 
            ofile.write(line)
        elif random.random() >= p:
            ofile.write(line)
    ifile.close()
    ofile.close()
    print 'changed'
    return 

def parser2(filename):
    articles = []
    words = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if 'trace' in line:
                if words: articles.append(words)
                words = []
            elif 'define' in line or line.startswith(')'): 
                continue
            else:
                line = line.replace('(', '').replace(')', '').strip()
                if not line: continue
                line = line.split(' ')
                words.extend(line)
    if words: articles.append(words)
    return articles

def NLP_Code(X, K=1):
    d = set([])
    newx = []
    for x in X:
        neww = []
        for ind in range(len(x)):
            if ind+K-1 < len(x):
                concatenated = '+'.join(map(str, x[ind:ind+K]))
                neww.append(concatenated)
        newx.append(neww)
        d.update(neww)

    d = list(d)
    arr = np.zeros(( len(newx), len(d) ))
    for ind, article in enumerate(newx):
        for word in article:
            arr[ind][d.index(word)] += 1
    return arr, d

def store2arff(X, Y, filename, dictionary, **kwargs):
    output = open(filename, 'w')
    output.write('@RELATION {}\n'.format(kwargs.get('RELATION', 'UNKNOWN')))
    for word in dictionary:
        output.write('@ATTRIBUTE {} NUMERIC\n'.format(word))
    output.write('@ATTRIBUTE SUCCESS {1, 0}\n')
    output.write('@DATA\n')
    for x, y in zip(X, Y):
#        output.write('{},{}\n'.format(','.join(map(lambda _: '1' if _>0 else '0', x)), y))
        output.write('{},{}\n'.format(','.join(map(str, x)), y))
    output.close()
    print '{} is ready ...'.format(filename)

def pca(x, variance_ratio=0.90):
    pca = PCA()
    pca.fit(x)
    acc = 0.0
    for ind, ele in enumerate(pca.explained_variance_ratio_):
        if acc > variance_ratio: break
        acc += ele
    ind += 1
    pca.set_params(n_components=ind)
    x = pca.fit_transform(x)
    print 'final shape: {}'.format(x.shape)
    return x


if __name__ == '__main__':   
    def lab(pfile, nfile, p):
        change(pfile, nfile, p)
        return parser2(pfile), parser2(nfile)

#    positive_x, negative_x = lab('./data/blocks-lib', './data/blocks-negative', p=0.20)
#    positive_x, negative_x = lab('./data/depots-lib', './data/depots-negative', p=0.20)
    positive_x, negative_x = lab('./data/driverlog-lib', './data/driverlog-negative', p=0.20)

    # features
    L1_x, dictionary = NLP_Code(positive_x+negative_x, K=1)
#   L2_x, dictionary = NLP_Code(positive_x+negative_x, K=2)
#   L2_x = pca(L2_x, 0.90) 
    x = np.concatenate([L1_x], 1)
    y = np.array(len(positive_x)*[1] + len(negative_x)*[0] )
    print 'x.shape={} y.shape={}'.format(x.shape, y.shape)

    # training
    svm = NuSVC(kernel='poly') # linear, poly, rbf, NuSVC
    lmnn = LMNN(k=5, learn_rate=1e-5, max_iter=200) 
    

    svmavr = []
    lmnnavr = []
    for _ in xrange(10):
        print 'Iteration {}'.format(_)

        svmrec = []
        lmnnrec  = []
        for train_index, test_index in KFold(len(x), n_folds=10, shuffle=True):
            train_x, test_x = x[train_index], x[test_index]
            train_y, test_y = y[train_index], y[test_index]

            svm.fit(train_x, train_y)
            svmrec.append( float((svm.predict(test_x) == test_y).sum())/ len(test_y) )

            L = lmnn.fit(train_x, train_y, verbose=True).L
            lmnnrec.append( knn(np.dot(train_x, L), train_y, np.dot(test_x, L), test_y) )

        print '\tSVM accuracy: {} = {}'.format(svmrec, np.mean(svmrec))
        svmavr.append(np.mean(svmrec))

        print '\tLMNN accuracy: {} = {}'.format(lmnnrec, np.mean(lmnnrec))
        lmnnavr.append(np.mean(lmnnrec))

    print 'SVM final accuracy: {}'.format(np.mean(svmavr))
    print 'LMNN final accuracy: {}'.format(np.mean(lmnnavr))


#    store2arff(x, y, 'parsed.arff', range(x.shape[1]))
