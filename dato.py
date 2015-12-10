#coding: utf8

import sqlite3
import graphlab as gl
from graphlab import SFrame
import argparse
import pickle as pkl


def iterReviewTrain(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    i = 0

    for pid, uname, rev, rating in c.execute('SELECT item as pid, user as uname, review as rev, rating as rating FROM reviews WHERE NOT test'):
        if pid is None or uname is None or rating is None or rev is None:
            continue
        else:
            yield (int(pid),int(uname),float(rating))

def iterReviewTest(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    i = 0

    for pid, uname, rev, rating in c.execute('SELECT item as pid, user as uname, review as rev, rating as rating FROM reviews WHERE test'):
        if pid is None or uname is None or rating is None or rev is None:
            continue
        else:
            yield (int(pid),int(uname),float(rating))



def main(args):
    all = [t for t in iterReviewTrain(args.db)]
    pid,uname,rating = zip(*all)

    train = {}
    train["users"] = uname
    train["items"]= pid
    train["ratings"] = rating
    train = SFrame(train)
    print(train)

    all = [t for t in iterReviewTest(args.db)]
    pid,uname,rating = zip(*all)

    test = {}
    test["users"] = uname
    test["items"]= pid
    test["ratings"] = rating
    test = SFrame(test)
    half, test = test.random_split(.5)
    print(test)


    params = dict([('user_id', 'users'),
                   ('item_id','items'),
                   ('target','ratings'),
                   ('num_factors', range(5,50)),
                   ('regularization', [0.0001,0.001,0.01,0.1,1]),
                   ('nmf',[True,False])
                   ])

    job = gl.random_search.create((train, test),
                                gl.factorization_recommender.create,
                                params)

    results = job.get_results()
    print(results)
    results.save('random_search', format='csv')


parser = argparse.ArgumentParser()
parser.add_argument("db", type=str)
args = parser.parse_args()

main(args)
