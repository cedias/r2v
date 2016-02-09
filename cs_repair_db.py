#coding: utf-8

import sqlite3
import argparse
from cytoolz import groupby
import pickle as pkl
import itertools
from cytoolz import keyfilter
from random import sample
import itertools

def iterReviews(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    for pid, uname, rev,rating in c.execute('SELECT item as pid, user as uname, review as rev, rating as rating FROM reviews'):
        yield (pid,uname,rating,rev)



def main(args):

    train,test = pkl.load(open(args.pkl,"rb"))
    key = next(iter(train.keys()))
    if train[key][0][0] == key:
        print("indexed by pid -> CS ITEM")
        t = 0
    else:
        print("indexed by uname -> CS USER")
        t = 1
    

    db = groupby(lambda x: x[t],iterReviews(args.db))
    big = keyfilter(lambda x:len(db[x])>=60,db) #whitelist
    
    
    test_set =  set(tup for tup in itertools.chain.from_iterable(test.values()))
    real_test = {k:sample([n for n in v if n not in test_set],30) for k,v in big.items()}

    
    pkl.dump((train,test,real_test),open(args.pkl,"wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str)
    parser.add_argument("pkl", type=str)
    args = parser.parse_args()

    main(args)