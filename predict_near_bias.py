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

def getUsersBias(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    c.execute(
        "SELECT user,avg(rating) as bias FROM reviews WHERE not test group by user")
    return c.fetchall()


def getItemsBias(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    c.execute(
        "SELECT item, avg(rating) as bias FROM reviews WHERE not test group by item")
    return c.fetchall()


def getFullBias(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    c.execute("SELECT avg(rating) as bias FROM reviews WHERE not test")
    return c.fetchone()[0]



def k_sim(model, db, test=True):

    print("prepping bias")
    u_bias = {user: bias for user, bias in getUsersBias(db)}
    i_bias = {item: bias for item, bias in getItemsBias(db)}
    db_avg = getFullBias(db)

    print("prepping data")
    test_data = [(matutils.unitvec(model["u_{}".format(user)] + model["i_{}".format(item)]) ,matutils.unitvec(model["u_{}".format(user)]) ,matutils.unitvec(model["i_{}".format(item)]) , u_bias[user], i_bias[item], float(rating)) for item, user, rating in getAllReviews(db, test=test) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
    rating_indexs = [(matutils.unitvec(model[word]), float(word.split("_")[1])) for word in model.index2word if len(word) > 2 and word[0] == "r" and word[1] == "_"]

    r_vec, ratings = zip(*rating_indexs)
    sum_vec,user_vec,item_vec, u_bias, i_bias, ground_truth = zip(*test_data)
    rand = [randint(0,len(ratings)-1)for i in range(0,len(test_data))]
    print("{} test data ready".format(len(test_data)))

    ratings = np.array(ratings)
    ground_truth = np.array(ground_truth)
    r_vec = np.array(r_vec).T
    sum_vec = np.array(sum_vec)
    user_vec = np.array(user_vec)
    item_vec = np.array(item_vec)
    u_bias = np.array(u_bias)
    i_bias = np.array(i_bias)


    dp = np.dot(sum_vec,r_vec)
    sum_args = np.argmax(dp, axis=1)

    dp = np.dot(user_vec,r_vec)
    user_args = np.argmax(dp, axis=1)

    dp = np.dot(item_vec,r_vec)
    item_args = np.argmax(dp, axis=1)


    avg = np.ones(len(ground_truth)) * db_avg
    user = ratings[user_args]
    item = ratings[item_args]
    ui_s = ratings[sum_args]



    err_avg = np.mean((avg - ground_truth) ** 2)
    err_user = np.mean((user - u_bias) ** 2)
    err_item = np.mean((item - i_bias) ** 2)
    err_sui = np.mean((ui_s - ground_truth) ** 2)

    print("avg {} + user {} + item {} + ui {}".format(err_avg,err_user,err_item,err_sui))
    print("{}".format((err_avg+err_sui)/2))
    
    print("Random Baseline")
    err = (ratings[rand] - ground_truth) ** 2
    print(np.mean(err))


parser = argparse.ArgumentParser()

parser.add_argument("model", type=str)
parser.add_argument("db", type=str)

args = parser.parse_args()
db = args.db
model = R2VModel.from_w2v_text(args.model,binary=True)
k_sim(model, db)
