import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from gensim import matutils
from random import shuffle

def getItemReviews(item, db):
    con = sqlite3.connect(db)
    c = con.cursor()
    c.execute("SELECT user,rating,review FROM reviews WHERE item = {} and not test".format(item))
    return c.fetchall()

def getUserReviews(user, db):
    con = sqlite3.connect(db)
    c = con.cursor()
    c.execute("SELECT item,rating,review FROM reviews WHERE user = {} and not test".format(user))
    return c.fetchall()

def getAllReviews(db,test=False):
    con = sqlite3.connect(db)
    c = con.cursor()
    if test:
        c.execute("SELECT item,user,rating FROM reviews WHERE test")
    else:
        c.execute("SELECT item,user,rating FROM reviews WHERE not test")
    return c.fetchall()

def getUsersBias(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    c.execute("SELECT user,avg(rating) as bias FROM reviews WHERE not test group by user")
    return c.fetchall()

def getItemsBias(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    c.execute("SELECT item, avg(rating) as bias FROM reviews WHERE not test group by item")
    return c.fetchall()



def k_sim(model, db,k=None,neigh="user",mean_norm=False):

    if neigh not in {"user","item"}:
        print("only {} as similarity".format(["user","item"]))

    print("prepping data")
    if mean_norm:
        if neigh == "user":
            u_bias = {user:bias for user,bias in getUsersBias(db)}
        elif neigh == "item":
            i_bias = {item:bias for item,bias in getItemsBias(db)}

    test_data = [(item, user, rating) for item, user, rating in getAllReviews(db, test=True)]
    shuffle(test_data)
    print("test data ready")

    cpt_test = 0
    cpt_skipped = 0
    err = 0

    for item, user, rating in test_data:

        if cpt_test >= len(test_data)/2: # we only evaluate on random 50%
            break

        if ("u_{}".format(user) not in model.vocab and neigh=="user") or ("i_{}".format(item) not in model.vocab and neigh=="item") or ("u_{}".format(user) not in model.vocab and "i_{}".format(item) not in model.vocab and neigh=="sum") : #skip not in vocab
            cpt_skipped += 1
            continue

        if neigh=="user":
            vect = model["u_{}".format(user)]
        elif neigh=="item":
            vect = model["i_{}".format(item)]
        elif neigh=="sum":
            vect = model["i_{}".format(item)] + model["u_{}".format(user)]
        else:
            raise Exception("Neigh not item nor user")

        vect = matutils.unitvec(vect)

        if neigh == "user":
            if mean_norm:
                list_sims = [(suser,srating-u_bias[suser],np.dot(matutils.unitvec(model["u_{}".format(suser)]),vect)) for suser,srating,_ in getItemReviews(item, db) if "u_{}".format(suser) in model.vocab]
            else:
                list_sims = [(suser,srating,np.dot(matutils.unitvec(model["u_{}".format(suser)]),vect)) for suser,srating,_ in getItemReviews(item, db) if "u_{}".format(suser) in model.vocab]

        elif neigh == "item":
            if mean_norm:
                list_sims = [(sitem,srating-i_bias[sitem],np.dot(matutils.unitvec(model["i_{}".format(sitem)]),vect)) for sitem,srating,_ in getUserReviews(user, db) if "i_{}".format(sitem) in model.vocab]
            else:
                list_sims = [(sitem,srating,np.dot(matutils.unitvec(model["i_{}".format(sitem)]),vect)) for sitem,srating,_ in getUserReviews(user, db) if "i_{}".format(sitem) in model.vocab]

        elif neigh == "sum":
                list_sims = [(suser,srating,np.dot(matutils.unitvec(model["u_{}".format(suser)]+model["i_{}".format(item)]),vect)) for suser,srating,_ in getItemReviews(item, db) if "u_{}".format(suser) in model.vocab]
                list_sims += [(sitem,srating,np.dot(matutils.unitvec(model["i_{}".format(sitem)]+model["u_{}".format(user)]),vect)) for sitem,srating,_ in getUserReviews(user, db) if "i_{}".format(sitem) in model.vocab]


        if len(list_sims) == 0:
            cpt_skipped += 1
            continue

        sim_users,sim_rating,sim_sim = zip(*list_sims)

        sim_sim = [(x+1)/(2+0.0) for x in sim_sim] # make sim between [0,1]
        order = np.argsort(sim_sim)[::-1]

        if len(order) == 0:
            cpt_skipped +=1
            continue
        
        if k is not None:    
            order = order[:k]

        
        sum_rs = sum([sim_rating[x]*sim_sim[x] for x in order])
        sum_sim = sum([sim_sim[x] for x in order])



        predicted = sum_rs/(sum_sim+0.0)

        if mean_norm:
            if neigh == "user":
                predicted +=  u_bias[user]
            elif neigh == "item":
                predicted +=  i_bias[item]


        err += (rating - predicted) ** 2

        cpt_test += 1
        if cpt_test % 100 == 0:
            print("MSE at {} tests is {} - {} test cases where skipped".format(cpt_test,err/(cpt_test+0.0),cpt_skipped))

    print("Final MSE for {} tests is {} - {} test cases where skipped".format(cpt_test,err/(cpt_test+0.0),cpt_skipped))


parser = argparse.ArgumentParser()

parser.add_argument("--k",default=5, type=int)
parser.add_argument("--neigh",default="item", type=str)
parser.add_argument('--mean_center', dest='mean_center', action='store_true')
parser.add_argument("model", type=str)
parser.add_argument("db", type=str)

args = parser.parse_args()
db = args.db
model = Doc2Vec.load_word2vec_format(args.model, binary=True,norm_only=False)
k_sim(model,db,k=args.k,neigh=args.neigh,mean_norm=args.mean_center)
