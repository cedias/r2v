import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from random import shuffle, randint
from sklearn import preprocessing, cross_validation, linear_model
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, mean_squared_error
from gensim import matutils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt





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


def getAllReviews(db, test=False):
    con = sqlite3.connect(db)
    c = con.cursor()
    if test:
        c.execute("SELECT item,user,rating FROM reviews WHERE test")
    else:
        c.execute("SELECT item,user,rating FROM reviews WHERE not test")
    return c.fetchall()


def get_dataset(model, db, norm, conc, vtarget, bias):
    print("prepping bias")

    if bias and not vtarget:
        u_bias = {user: bias for user, bias in getUsersBias(db)}
        i_bias = {item: bias for item, bias in getItemsBias(db)}
        db_avg = getFullBias(db)

    if bias:
        if not vtarget:
            if not conc:
                test_data = [(model["u_{}".format(user)] + model["i_{}".format(item)], float(rating), i_bias[item], u_bias[user]) for item, user,
                             rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
            else:
                test_data = [(np.concatenate((model["u_{}".format(user)], model["i_{}".format(item)])), float(rating), i_bias[item], u_bias[user]) for item, user,
                             rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
    else:
        if not vtarget:
            if not conc:
                test_data = [(model["u_{}".format(user)] + model["i_{}".format(item)], float(rating)) for item, user,
                             rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
            else:
                test_data = [(np.concatenate((model["u_{}".format(user)], model["i_{}".format(item)])), float(rating)) for item, user,
                             rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
        else:
            if not conc:
                test_data = [(model["u_{}".format(user)] + model["i_{}".format(item)], model["r_" + str(rating)]) for item, user,
                             rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
            else:
                test_data = [(np.concatenate((model["u_{}".format(user)], model["i_{}".format(item)])), model["r_" + str(rating)]) for item,
                             user, rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]

    if bias:
        test_vec, ground_truth, item_b, user_b = zip(*test_data)
        test_vec = np.array(test_vec)
        ground_truth = np.array(ground_truth)
        item_b = np.array(item_b)
        user_b = np.array(user_b)
        ground_truth = ground_truth - \
            (db_avg + (item_b - db_avg) + (user_b - db_avg))
    else:
        test_vec, ground_truth = zip(*test_data)
        test_vec = np.array(test_vec)
        ground_truth = np.array(ground_truth)

    if norm:
        test_vec = preprocessing.normalize(test_vec, copy=False)

    print("{} test data".format(len(test_data)))

    rd = np.random.rand(len(test_data))
    rd = rd > 0.5
    train_vectors = test_vec[rd]
    train_labels = ground_truth[rd]

    print("{} vecteurs d'apprentissage".format(len(train_vectors)))
    test_vectors = test_vec[np.logical_not(rd)]
    test_labels = ground_truth[np.logical_not(rd)]

    print("{} vecteurs de test".format(len(train_vectors)))
    return (train_vectors, train_labels, test_vectors, test_labels)


def get_multi_score(clf, x, y, vtarget):
    pred = clf.predict(x)

    if not vtarget:
        mse = mean_squared_error(y, pred)

    else:
        rating_indexs = [(matutils.unitvec(model[word]), float(word.split("_")[1]))
                         for word in model.index2word if len(word) > 2 and word[0] == "r" and word[1] == "_"]
        r_vec, ratings = zip(*rating_indexs)
        ratings = np.array(ratings)
        r_vec = np.array(r_vec).T
        dp = np.dot(pred, r_vec)
        dp2 = np.dot(y, r_vec)
        print(np.argmax(dp, axis=1))
        err = (ratings[np.argmax(dp, axis=1)] -
               ratings[np.argmax(dp2, axis=1)]) ** 2
        mse = np.mean(err)

    return mse


def k_sim(model, db, norm=False, conc=False, vtarget=False, bias=False):

    mlp = Sequential()
    mlp.add(Dense(output_dim=400, input_dim=400))
    mlp.add(Activation('linear'))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(output_dim=1, input_dim=400, W_regularizer=l2(0.01)))
    mlp.compile(loss='mean_squared_error', optimizer='adagrad')

    train_vectors, train_labels, test_vectors, test_labels = get_dataset(
        model, db, norm, conc, vtarget, bias)

    hist = mlp.fit(train_vectors, train_labels, validation_split=0.2, nb_epoch=50, batch_size=16)
    score = mlp.evaluate(test_vectors, test_labels, batch_size=16)
    print(score)
    plt.plot(hist)


parser = argparse.ArgumentParser()

parser.add_argument("model", type=str)
parser.add_argument("db", type=str)
parser.add_argument("--norm", dest="norm", action="store_true")
parser.add_argument("--concat", dest="concat", action="store_true")
parser.add_argument("--vec_target", dest="vec_target", action="store_true")
parser.add_argument("--bias", dest="bias", action="store_true")

args = parser.parse_args()
db = args.db
norm = args.norm
conc = args.concat
bias = args.bias
vtarget = args.vec_target
model = Doc2Vec.load_word2vec_format(args.model, binary=True, norm_only=False)
k_sim(model, db, norm, conc, vtarget, bias)
