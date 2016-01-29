#coding: utf8

import sqlite3
from random import shuffle
import argparse
from cytoolz import frequencies, keyfilter
import pickle as pkl
import itertools


def level(csdb,level):
    train,test = csdb
    if level < 1:
        return train, test

    add_to_train = {k:v[:level] for k,v in test.items()}    

    for k,v in test.items():
        del(v[:level])

    train.update(add_to_train)
    return train,test

def filterWords(words, min_count=0):
    return set(word for word in words if words[word] >= min_count)


def iterReviews(train):

    for pid,uname, rating , rev in itertools.chain.from_iterable(train.values()):
        if pid is None or uname is None or rating is None or rev is None:
            continue
        else:
            for sent in rev.split("."):
                if len(sent) < 2:
                    continue
                else:
                    yield (sent.split(" "), ['u_{}'.format(uname),'i_{}'.format(pid), 'r_{}'.format(rating)])
                   


def main(args):
    print("output is {}".format(args.output)) 
    f = open(args.output, "w",encoding="utf-8")
    train,test = level(pkl.load(open(args.csdb,"rb")), args.level) 
    buff = []
    i = 0

    wf= frequencies(itertools.chain.from_iterable([rev.split() for _,_,_,rev in itertools.chain.from_iterable(train.values())]))
    words = keyfilter(lambda k: wf[k] >args.min_count,  wf)


    for sent, labels in iterReviews(train):
        sent = [word for word in sent if word in words]

        if len(sent) < args.min_sent_size:
            continue

        for label in labels:
            buff.append("{} {}\n".format(label, " ".join(sent)))
            i += 1

        if len(buff) >= args.buff_size:
            shuffle(buff)
            for se in buff:
                f.write(se)
            buff = []
            print("wrote {} sentences".format(i))

    shuffle(buff)
    for se in buff:
        f.write(se)
    f.close()

    print("wrote {} sentences".format(i))

parser = argparse.ArgumentParser()
parser.add_argument("csdb", type=str)
parser.add_argument("level", type=int)
parser.add_argument("output", type=str, default="sentences.txt")
parser.add_argument("--min_count", type=int, default=100)
parser.add_argument("--min_sent_size", type=int, default=5)
parser.add_argument("--buff_size", type=int, default=1000000)
args = parser.parse_args()

main(args)
