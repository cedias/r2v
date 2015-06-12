import numpy as np
import argparse
from os import path
from random import shuffle


class Node(object):

    """Class that implement the node of the parsing tree"""

    def __init__(self, word=None, label=None):
        self.y = label
        if label is not None:
            self.ypred = np.ones(len(label)) / len(label)
        else:
            self.ypred = None
        self.X = None
        self.word = word
        self.parent = None
        self.childrens = []
        self.d = None  # Vecteur d'erreur


class Tree(object):

    """docstring for Tree"""

    def __init__(self, sentence, structure, label=None):
        self.sentence = sentence
        self.structure = structure

        wc = len(sentence)

        self.nodes = []
        self.leaf = []

        for i in range(2 * wc - 1):
            self.nodes.append(Node())

        for i, w in enumerate(sentence):
            node = self.nodes[i]
            node.word = w
            node.order = i
            self.leaf.append(node)

        parc = {}
        for i, (n, p) in enumerate(zip(self.nodes, structure)):
            n.parent = p - 1
            self.nodes[p - 1].childrens.append(i)
            l = parc.get(p - 1, [])
            l.append(i)
            parc[p - 1] = l
        parc.pop(-1)
        self.parcours = list(parc.items())
        self.parcours.sort()

        if label is not None:
            for n in self.leaf:
                n.y = np.zeros(2)
                n.y[1] = label[n.word]
                n.y[0] = 1 - n.y[1]

            for p, [a, b] in self.parcours:
                aT = self.nodes[a]
                bT = self.nodes[b]
                pT = self.nodes[p]
                if aT.order < bT.order:
                    pT.word = ' '.join([aT.word, bT.word])
                else:
                    pT.word = ' '.join([bT.word, aT.word])
                pT.y = np.zeros(2)
                pT.y[1] = label[pT.word]
                pT.y[0] = 1 - pT.y[1]
                pT.order = aT.order

    @staticmethod
    def getSoftLabel(l):
        if l <= 0.2:
            label = 0
        elif 0.2 < l <= 0.4:
            label = 1
        elif 0.4 < l <= 0.6:
            label = 2
        elif 0.6 < l <= 0.8:
            label = 3
        else:
            label = 4
        return label


def load(datafolder):
    print('Load Trees...')
    with open(path.join(datafolder, 'STree.txt')) as f:
        trees = []
        for line in f.readlines():
            tree = line.split('|')
            tree = np.array(tree).astype(int)
            trees.append(tree)

    print('Load Sentences...')
    with open(path.join(datafolder, 'SOStr.txt')) as f:
        sentences = []
        lexicon = set()
        for line in f.readlines():
            sent = line.strip().split('|')
            sentences.append(sent)
            lexicon = lexicon.union(sent)

    print('Load data split')
    with open(path.join(datafolder, 'datasetSplit.txt')) as f:
        whichSet = []
        f.readline()
        for line in f.readlines():
            whichSet.append(int(line.strip().split(',')[1]))

    print('Load Index...')
    with open(path.join(datafolder, 'dictionary.txt')) as f:
        index = {}
        for line in f.readlines():
            phrase = line.split('|')
            index[int(phrase[1])] = phrase[0]

    print('Load Labels...')
    with open(path.join(datafolder, 'sentiment_labels.txt')) as f:
        f.readline()
        labels = {}
        for line in f.readlines():
            id_p, y = line.split('|')
            labels[index[int(id_p)]] = float(y)

    print('Build Trees...')
    X_trees_train = []
    X_trees_dev = []
    X_trees_test = []
    for s, t, k in zip(sentences, trees, whichSet):
        if k == 1:
            X_trees_train.append(Tree(s, t, labels))
        elif k == 2:
            X_trees_test.append(Tree(s, t, labels))
        elif k == 3:
            X_trees_dev.append(Tree(s, t, labels))
        else:
            raise(Exception('Erreur dans le parsing train/test/dev'))
    return lexicon, X_trees_train, X_trees_dev, X_trees_test, labels


def main(args):
    l, train, dev, test, lab = load(args.datafolder)
    sentences = []
    classes = args.classes
    fs = args.full_sentences
    known = 0
    for tree in train:
        sent = " ".join(tree.sentence)
        label = Tree.getSoftLabel(lab[sent])

        if classes == 2:
            if label == 2:
                continue
            elif label < 2:
                label = 0
            elif label > 2:
                label = 1

        sentences.append("sent_{} {}\n".format(label, sent))

        if fs:
            sentences.append("kn_{}_{} {}\n".format(label,known, sent))
            known += 1


    for tree in dev:
        sent = " ".join(tree.sentence)
        label = Tree.getSoftLabel(lab[sent])

        if classes == 2:
            if label == 2:
                continue
            elif label < 2:
                label = 0
            elif label > 2:
                label = 1

        sentences.append("sent_{} {}\n".format(label, sent))

        if fs:
            sentences.append("kn_{}_{} {}\n".format(label,known, sent))
            known += 1

    for i,tree in enumerate(test):
        sent = " ".join(tree.sentence)
        label = Tree.getSoftLabel(lab[sent])

        if classes == 2:
            if label == 2:
                continue
            elif label < 2:
                label = 0
            elif label > 2:
                label = 1

        sentences.append("unk_{}_{} {}\n".format(label,i, sent))

    shuffle(sentences)

    with open(args.output,"w") as outfile:
        for s in sentences:
            outfile.write(s)


parser = argparse.ArgumentParser()
parser.add_argument("datafolder",default="Data", type=str)
parser.add_argument("--output",default="treebank.d2v", type=str)
parser.add_argument("--classes",default=5,type=int)
parser.add_argument("--full_sentences",default=False,type=bool)
args = parser.parse_args()


if args.classes != 5 and args.classes != 2:
    print("classes is 5 or 2")
else:
    main(args)
