import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from random import shuffle,randint
from gensim import matutils
from VectReco.R2VModel import R2VModel


def getAllReviews(db, test=False):
    con = sqlite3.connect(db)
    c = con.cursor()
    if test:
        c.execute("SELECT item,user,rating FROM reviews WHERE test")
    else:
        c.execute("SELECT item,user,rating FROM reviews WHERE not test")
    return c.fetchall()


def k_sim(model, db,test=True):

    #print("prepping data")

  
    test_data = [(matutils.unitvec(model["u_{}".format(user)] + model["i_{}".format(item)])  , float(rating)) for item, user, rating in getAllReviews(db, test=test) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
    rating_indexs = [(matutils.unitvec(model[word]), float(word.split("_")[1])) for word in model.index2word if len(word) > 2 and word[0] == "r" and word[1] == "_"]
    r_vec, ratings = zip(*rating_indexs)
    sum_vec, ground_truth = zip(*test_data)

    #print("{} data ready".format(len(test_data)))

    ratings = np.array(ratings)
    ground_truth = np.array(ground_truth)
    r_vec = np.array(r_vec).T
    sum_vec = np.array(sum_vec)

    dp = np.dot(sum_vec,r_vec)
    sum_args = np.argmax(dp, axis=1)

    diff = ratings[sum_args] - ground_truth
    mse = np.mean((diff) ** 2)
    diff[np.where(diff != 0)] = 1
    acc = (len(diff) - np.sum(diff))/len(diff) * 100

    return(mse,acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str)
    parser.add_argument("db", type=str)

    args = parser.parse_args()
    db = args.db
    model = R2VModel.from_w2v_text(args.model,binary=True)

    mse_train, acc_train = k_sim(model, db,False)
    mse_test, acc_test = k_sim(model, db,True)
    print("mse: {:.3f} acc: {:.3f}% (train) \nmse: {:.3f} acc: {:.3f}% (test)".format(mse_train,acc_train,mse_test,acc_test))
