import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from random import shuffle,randint
from gensim import matutils



def getAllReviews(db, test=False):
    con = sqlite3.connect(db)
    c = con.cursor()
    if test:
        c.execute("SELECT item,user,rating FROM reviews WHERE test")
    else:
        c.execute("SELECT item,user,rating FROM reviews WHERE not test")
    return c.fetchall()


def k_sim(model, db,pond=True,solo=False):

    print("prepping data")

    test_data = [(matutils.unitvec(model["u_{}".format(user)] + model["i_{}".format(item)]) ,matutils.unitvec(model["u_{}".format(user)]) ,matutils.unitvec(model["i_{}".format(item)]) , float(rating)) for item, user, rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]

    rating_indexs = [(matutils.unitvec(model[word]), float(word.split("_")[1])) for word in model.index2word if len(word) > 2 and word[0] == "r" and word[1] == "_"]

    r_vec, ratings = zip(*rating_indexs)
    sum_vec,user_vec,item_vec, ground_truth = zip(*test_data)

    rand = [randint(0,len(ratings)-1)for i in range(0,len(test_data))]
    print("{} test data ready".format(len(test_data)))

    ratings = np.array(ratings)
    ground_truth = np.array(ground_truth)
    r_vec = np.array(r_vec).T
    sum_vec = np.array(sum_vec)
    user_vec = np.array(user_vec)
    item_vec = np.array(item_vec)

    if solo:
        dp = np.dot(sum_vec,r_vec)
        sum_args = np.argmax(dp, axis=1)
        print("User + Item = Rating")
        err = (ratings[sum_args] - ground_truth) ** 2
        print(np.mean(err))

    elif pond:
        dp = np.dot(sum_vec,r_vec)

        res = ratings[np.argmax(dp, axis=1)]*np.max(dp,axis=1)
        sum_max  = np.max(dp,axis=1)


        dp = np.dot(user_vec,r_vec)
        res += ratings[np.argmax(dp, axis=1)] * np.max(dp,axis=1)
        sum_max  += np.max(dp,axis=1)

        dp = np.dot(item_vec,r_vec)
        res += ratings[np.argmax(dp, axis=1)]* np.max(dp,axis=1)
        sum_max  += np.max(dp,axis=1)


        print("User rating + Item rating  + (user + item) rating = Rating - Cos Sim ponderation")
        err = ((res/sum_max) - ground_truth) ** 2
        print(np.mean(err))

    else:

        dp = np.dot(sum_vec,r_vec)
        sum_args = np.argmax(dp, axis=1)

        dp = np.dot(user_vec,r_vec)
        user_args = np.argmax(dp, axis=1)

        dp = np.dot(item_vec,r_vec)
        item_args = np.argmax(dp, axis=1)


        print("User rating + Item rating  + (user + item) rating = Rating - no ponderation")

        err = ((ratings[sum_args]+ratings[user_args]+ratings[item_args])/3 - ground_truth) ** 2
        print(np.mean(err))

    print("Random Baseline")
    err = (ratings[rand] - ground_truth) ** 2
    print(np.mean(err))


parser = argparse.ArgumentParser()

parser.add_argument("model", type=str)
parser.add_argument("db", type=str)
parser.add_argument("--pond",dest="pond",action="store_true")
parser.add_argument("--solo",dest="solo",action="store_true")

args = parser.parse_args()
db = args.db
pond = args.pond
model = Doc2Vec.load_word2vec_format(args.model, binary=True,norm_only=False)
k_sim(model, db,pond)
