#coding: utf-8

import sqlite3
import argparse
from cytoolz import groupby
import pickle as pkl
import itertools
from cytoolz import keyfilter
from random import sample

def iterReviews(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    for pid, uname, rev,rating in c.execute('SELECT item as pid, user as uname, review as rev, rating as rating FROM reviews'):
        yield (pid,uname,rating,rev)


def main(args):
    t = args.type

    db = groupby(lambda x: x[t],iterReviews(args.db))

    big = keyfilter(lambda x:len(db[x])>=60,db) #whitelist
    test = {k:sample(v,30)for k,v in big.items()}
    train = keyfilter(lambda x:x not in big, db)


    print(len(big))
    print(len(train))
    
    pkl.dump((train,test),open(args.output,"wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str)
    parser.add_argument("--type",type=int,default=1)
    parser.add_argument("--output", type=str, default="db.pkl")
    args = parser.parse_args()

    if args.type != 0 and args.type != 1:
        print("type = 0 for item CS and 1 for user CS [default to user CS (1)]")
        args.type = 1

    main(args)