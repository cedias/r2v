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

    RSE = {"jaccard":[],"cosine":[],"pearson":[]}

    for method in ["jaccard","cosine","pearson"]:
        for k in xrange(1,25):
            sim = gl.recommender.item_similarity_recommender.create(train,user_id="users",item_id="items",similarity_type=method,only_top_k=k,target="ratings",verbose=False)
            RSE[method].append(sim.evaluate_rmse(test,target="ratings")["rmse_overall"])
        
    for method in ["jaccard","cosine","pearson"]:
        plt.plot(range(1,25),RSE[method],label=method)

    pkl.dump(RSE,open(args.output,'wb'))


parser = argparse.ArgumentParser()
parser.add_argument("db", type=str)
parser.add_argument("output",type=str)
args = parser.parse_args()

main(args)
