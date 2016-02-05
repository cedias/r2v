#coding: utf-8

import argparse
from gensim.models.word2vec import Word2Vec
import numpy as np
from gensim import matutils
from sklearn.preprocessing import normalize
from cytoolz import groupby
import pickle as pkl
import itertools
from tqdm import tqdm




def k_sim(model, db,k=None,):

    train,test = db
    #(pid,uname,rating,rev)

    print("calcul moyenne générale train:")
    mu = np.mean([x[2] for x in itertools.chain.from_iterable(train.values())])
    print("mu is {}".format(mu))

    key = next(iter(train.keys()))

    if train[key][0][0] == key:
        print("indexed by pid -> CS ITEM")
        type_pred = "item"
        ui_dic =  dict(groupby(lambda x:x[1],itertools.chain.from_iterable(train.values())))
        iu_dic = train
    else:
        print("indexed by uname -> CS USER")
        type_pred = "user"
        iu_dic = dict(groupby(lambda x:x[0],itertools.chain.from_iterable(train.values())).items())
        ui_dic = train


    cpt_test = 0
    cpt_skipped = 0
    tot_err = np.zeros(k)

    err_mean = 0

    for pid,uname,rating,rev in tqdm(list(itertools.chain.from_iterable(test.values()))):




        try:

            if type_pred=="user":
                vect = model["u_{}".format(uname)]
            else:
                vect = model["i_{}".format(pid)]


            vect = matutils.unitvec(vect)

            if type_pred == "user" :
                list_sims = [(suser,srating-mu,model["u_{}".format(suser)]) for _,suser,srating,_ in iu_dic[pid]]
            else:
                list_sims = [(sitem,srating-mu,model["i_{}".format(sitem)]) for sitem,_,srating,_ in ui_dic[uname]]
        except:
            cpt_skipped += 1
            continue



        sim_users,sim_rating,sim_sim = zip(*list_sims)
        sim_rating = np.array(sim_rating)
        sim_sim = np.array(sim_sim)

        sim_sim = normalize(sim_sim, copy=False)
        sim_sim = np.dot(vect,sim_sim.T)

        sim_sim += 1
        sim_sim /= 2.0
        order = np.argsort(sim_sim)[::-1]



        order = np.array(order)
        sim_sim = sim_sim[order]
        sim_rating = sim_rating[order]

        sim_sim = sim_sim[:k]
        sim_rating = sim_rating[:k]


        pond = sim_rating * sim_sim

        sum_rs = np.cumsum(pond)
        sum_sim = np.cumsum(sim_sim)


        predicted = sum_rs/(sum_sim+0.0)


        predicted += mu


        err = (rating - predicted) ** 2
        err_mean += (rating - mu) ** 2

        if len(err) != k:
            oldlen = len(err)
            err.resize(k,refcheck=False)
            err[oldlen:k] = err[oldlen-1]

        tot_err += err
        cpt_test += 1


    print("Final MSE for {} tests is {} - {} test cases where skipped - mean = {}".format(cpt_test,tot_err/(cpt_test+0.0),cpt_skipped,err_mean/cpt_test))
    return tot_err/(cpt_test+0.0)


parser = argparse.ArgumentParser()

parser.add_argument("--k",default=5, type=int)
parser.add_argument('--output', default="predicted_ratings.pkl",type=str)
parser.add_argument("model", type=str)
parser.add_argument("db", type=str)



args = parser.parse_args()
model = Word2Vec.load_word2vec_format(args.model, binary=True)
db = pkl.load(open(args.db,"rb"))


err = k_sim(model,db, k=args.k)

result = {
    "model":args.model,
    "db":args.db,
    "error":err
}

pkl.dump(result,open(args.output,"wb"))
