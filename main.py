import os
import random
import cPickle
import numpy as np

import sys

from sklearn.decomposition import PCA
from sklearn.svm import SVC, NuSVC
from sklearn.cross_validation import train_test_split, KFold

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


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
        else:
            continue
            # delete pass to change "deletion" to "shuffling" 
            components = line.replace('(', '').replace(')', '').strip().split(' ')
#           print components
            action = components.pop(0)
            np.random.shuffle(components)
            components = [action] + components 
#           print components
            ofile.write(' '.join(components))

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
    return x, pca

def knn(train_x, train_y, test_x, test_y, K=5):
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(train_x, train_y)
    acc = (neigh.predict(test_x) == test_y).sum()
    return float(acc)/test_y.shape[0]


if __name__ == '__main__':   
    try:
        ind = sys.argv.index('-max')    
        MAX = int(sys.argv[ind+1])
    except:
        print 'cannot find max argument'
        print 'set MAX as 400'
        MAX = 400


    def lab(pfile, nfile, p):
        change(pfile, nfile, p)
        return parser2(pfile), parser2(nfile)

    positive_x, negative_x = lab('./data/blocks-lib', './data/blocks-negative', p=0.40)
#    positive_x, negative_x = lab('./data/depots-lib', './data/depots-negative', p=0.20)
#    positive_x, negative_x = lab('./data/driverlog-lib', './data/driverlog-negative', p=0.20)

    # features
#    L1_x, dictionary = NLP_Code(positive_x+negative_x, K=1)
#    L2_x, dictionary = NLP_Code(positive_x+negative_x, K=2)
#    L4_x, dictionary = NLP_Code(positive_x+negative_x, K=4)

#    L2_x, _ = pca(L2_x, 0.99) 
#    L4_x, _ = pca(L4_x, 0.99) 

    xs = []
    for l in ['-1', '-2','-3','-4']:
        if l in sys.argv:
            l = int(l[1])

            _x, dictionary = NLP_Code(positive_x+negative_x, K=l)
            _x, _ = pca(_x, 0.99) 
            xs.append(_x)
            print 'Adding level-{} data'.format(l)

    x = np.concatenate(xs, 1)
    y = np.array(len(positive_x)*[1] + len(negative_x)*[0] )
    print 'x.shape={} y.shape={}'.format(x.shape, y.shape)

    index = np.random.permutation(len(x))
    x = x[index]
    y = y[index]
#   x -= x.min(1).reshape(-1, 1)
#   x /= x.max(1).reshape(-1, 1)

    # truncate
    x = x[:MAX, :]
    y = y[:MAX]
    print 'MAX={}'.format(MAX)
    sys.stdout.flush()


    # training
    svm = NuSVC(kernel='linear') # linear, poly, rbf, NuSVC
    lmnn = LMNN(k=5, learn_rate=1e-7, max_iter=400) 
    gnb = GaussianNB()
    mnb = MultinomialNB(alpha=0.0)
    bnb = BernoulliNB(alpha=0.0)

    svmrec = []
    lmnnrec  = []
    gnbrec = []
    mnbrec = []
    bnbrec = []

    for train_index, test_index in KFold(len(x), n_folds=10, shuffle=True):
        train_x, test_x = x[train_index], x[test_index]
        train_y, test_y = y[train_index], y[test_index]

        gnb.fit(train_x, train_y)
        gnbrec.append( float((gnb.predict(test_x) == test_y).sum())/ len(test_y) )

        nonneg_train_x = train_x - train_x.min()
        nonneg_test_x = test_x - test_x.min()
        mnb.fit(nonneg_train_x, train_y)
        mnbrec.append( float((mnb.predict(nonneg_test_x) == test_y).sum())/ len(test_y) )

        bnb.fit(train_x, train_y)
        bnbrec.append( float((bnb.predict(test_x) == test_y).sum())/ len(test_y) )

        svm.fit(train_x, train_y)
        svmrec.append( float((svm.predict(test_x) == test_y).sum())/ len(test_y) )

        _ = PCA(n_components=20).fit(train_x)
        train_x = _.transform(train_x)
        test_x = _.transform(test_x)
        print train_x.shape
        L = lmnn.fit(train_x, train_y, verbose=True).L
        lmnnrec.append( knn(np.dot(train_x, L), train_y, np.dot(test_x, L), test_y, K=5) )

    print '\tSVM accuracy: {} = {}'.format(svmrec, np.mean(svmrec))
    print '\tLMNN accuracy: {} = {}'.format(lmnnrec, np.mean(lmnnrec))
    print '\tGaussianNB accuracy: {} = {}'.format(gnbrec, np.mean(gnbrec))
    print '\tMultinomiaNB accuracy: {} = {}'.format(mnbrec, np.mean(mnbrec))
    print '\tBernoulliNB accuracy: {} = {}'.format(bnbrec, np.mean(bnbrec))



#   lmnnavr.append(np.mean(lmnnrec))
#   gnbavr.append(np.mean(gnbrec))
#   svmavr.append(np.mean(svmrec))
#   svmavr = []
#   lmnnavr = []
#   gnbavr = []
#   print 'GuassianNB final accuracy: {}'.format(np.mean(gnbavr))
#   print 'SVM final accuracy: {}'.format(np.mean(svmavr))
#   print 'LMNN final accuracy: {}'.format(np.mean(lmnnavr))


#    store2arff(x, y, 'parsed{}.arff'.format(MAX), range(x.shape[1]))
