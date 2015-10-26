import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from random import shuffle,randint
from sklearn import preprocessing,cross_validation,linear_model
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, mean_squared_error


def getAllReviews(db, test=False):
    con = sqlite3.connect(db)
    c = con.cursor()
    if test:
        c.execute("SELECT item,user,rating FROM reviews WHERE test")
    else:
        c.execute("SELECT item,user,rating FROM reviews WHERE not test")
    return c.fetchall()


def get_dataset(model,db,norm,conc,vtarget):
    print("prepping data")

    if not vtarget:
        if not conc:
            test_data = [(model["u_{}".format(user)] + model["i_{}".format(item)] , float(rating)) for item, user, rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
        else:
            test_data = [(np.concatenate((model["u_{}".format(user)],model["i_{}".format(item)])) , float(rating)) for item, user, rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
    else:
        if not conc:
            test_data = [(model["u_{}".format(user)] + model["i_{}".format(item)] , model["r_"+str(rating)]) for item, user, rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
        else:
            test_data = [(np.concatenate((model["u_{}".format(user)],model["i_{}".format(item)])) , model["r_"+str(rating)]) for item, user, rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]


    test_vec, ground_truth = zip(*test_data)
    test_vec = np.array(test_vec)
    ground_truth = np.array(ground_truth)

    if norm:
        test_vec = preprocessing.normalize(test_vec,copy=False)

    print("{} test data".format(len(test_data)))

    rd=np.random.rand(len(test_data))
    rd=rd>0.5
    train_vectors=test_vec[rd]
    train_labels=ground_truth[rd]

    print("{} vecteurs d'apprentissage".format(len(train_vectors)))
    test_vectors=test_vec[np.logical_not(rd)]
    test_labels=ground_truth[np.logical_not(rd)]

    print("{} vecteurs de test".format(len(train_vectors)))
    return (train_vectors,train_labels,test_vectors,test_labels)


def get_multi_score(clf,x,y):
    pred = clf.predict(x)
    mse = mean_squared_error(y,pred)

    return mse



def k_sim(model, db,norm=False,conc=False,vtarget=False):

    train_vectors,train_labels,test_vectors,test_labels =  get_dataset(model,db,norm,conc,vtarget)



    clf = linear_model.LinearRegression()
    clf.fit(train_vectors,train_labels)

    print("TRAIN MSE: {}".format(get_multi_score(clf,train_vectors,train_labels)))
    print("TEST  MSE: {}".format(get_multi_score(clf,test_vectors,test_labels)))


    for c in np.arange(0.1,2,0.1):

        clf = linear_model.Ridge(alpha=c)
        clf.fit(train_vectors,train_labels)

        print("TRAIN MSE: {}".format(get_multi_score(clf,train_vectors,train_labels)))
        print("TEST  MSE: {}".format(get_multi_score(clf,test_vectors,test_labels)))




parser = argparse.ArgumentParser()

parser.add_argument("model", type=str)
parser.add_argument("db", type=str)
parser.add_argument("--norm",dest="norm",action="store_true")
parser.add_argument("--concat",dest="concat",action="store_true")
parser.add_argument("--vec_target",dest="vec_target",action="store_true")

args = parser.parse_args()
db = args.db
norm = args.norm
conc = args.concat
vtarget = args.vec_target
model = Doc2Vec.load_word2vec_format(args.model, binary=True,norm_only=False)
k_sim(model, db,norm,conc,vtarget)
