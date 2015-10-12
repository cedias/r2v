import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from random import shuffle,randint
from sklearn import preprocessing,cross_validation,linear_model


def getAllReviews(db, test=False):
    con = sqlite3.connect(db)
    c = con.cursor()
    if test:
        c.execute("SELECT item,user,rating FROM reviews WHERE test")
    else:
        c.execute("SELECT item,user,rating FROM reviews WHERE not test")
    return c.fetchall()


def k_sim(model, db,norm=False):

    print("prepping data")

    test_data = [(model["u_{}".format(user)] + model["i_{}".format(item)] , float(rating)) for item, user, rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]

    test_vec, ground_truth = zip(*test_data)
    label_encoder = preprocessing.LabelEncoder()
    ground_truth = np.array(ground_truth)
    ground_truth = label_encoder.fit_transform(ground_truth)
    print("{} test data ready".format(len(test_data)))

    if norm:
        test_vec = preprocessing.normalize(test_vec,copy=False)

    clf = linear_model.LogisticRegression(C=1e5)
    scores = cross_validation.cross_val_score(clf, test_vec, ground_truth, cv=5,scoring="accuracy")
    print(scores)



parser = argparse.ArgumentParser()

parser.add_argument("model", type=str)
parser.add_argument("db", type=str)
parser.add_argument("--norm",dest="norm",action="store_true")

args = parser.parse_args()
db = args.db
norm = args.norm
model = Doc2Vec.load_word2vec_format(args.model, binary=True,norm_only=False)
k_sim(model, db,norm)
