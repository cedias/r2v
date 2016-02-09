#coding: utf-8

import argparse
#from gensim.models.word2vec import Word2Vec
import numpy as np
#from gensim import matutils
from sklearn.preprocessing import normalize
from cytoolz import groupby
import pickle as pkl
import itertools
from tqdm import tqdm




def k_sim(db):

    train,_,test = db
    #(pid,uname,rating,rev)

    print("calcul moyenne générale train:")
    mu = np.mean([x[2] for x in itertools.chain.from_iterable(train.values())])
    print("mu is {}".format(mu))

    key = next(iter(train.keys()))

    if train[key][0][0] == key:
        print("indexed by pid -> CS ITEM")
        type_pred = "item"
        ui_dic =  dict(groupby(lambda x:x[1],itertools.chain.from_iterable(train.values())))
        um = { k: np.mean([r for _,_,r,_ in v]) for k,v in ui_dic.items()}
        pred = np.array([um[uname] for _,uname,_,_ in tqdm(list(itertools.chain.from_iterable(test.values()))) if uname in um])
        truth = np.array([rating for _,uname,rating,_ in tqdm(list(itertools.chain.from_iterable(test.values()))) if uname in um])



    else:
        print("indexed by uname -> CS USER")
        type_pred = "user"
        iu_dic = dict(groupby(lambda x:x[0],itertools.chain.from_iterable(train.values())).items())
        im = { k: np.mean([r for _,_,r,_ in v]) for k,v in iu_dic.items()}
        pred = np.array([im[pid] for pid,_,_,_ in tqdm(list(itertools.chain.from_iterable(test.values()))) if pid in im])
        truth = np.array([rating for pid,_,rating,_ in tqdm(list(itertools.chain.from_iterable(test.values()))) if pid in im])


    mean_pred = np.ones(len(truth))*mu
    mse = np.mean((pred - truth) **2)
    mean_mse = np.mean((mean_pred - truth) **2)
    pred
    if type_pred is "user":
        mn = "item"
    else:
        mn = "user"

    print("mean: {} {}-mean: {}".format(mean_mse,mn,mse))


parser = argparse.ArgumentParser()

parser.add_argument('--output', default="predicted_ratings.pkl",type=str)
parser.add_argument("db", type=str)



args = parser.parse_args()
db = pkl.load(open(args.db,"rb"))


err = k_sim(db)

result = {
    "db":args.db,
    "error":err
}

pkl.dump(result,open(args.output,"wb"))
