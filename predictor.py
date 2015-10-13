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


def get_dataset(model,db,norm):
    print("prepping data")
    test_data = [(model["u_{}".format(user)] + model["i_{}".format(item)] , float(rating)) for item, user, rating in getAllReviews(db, test=True) if "u_{}".format(user) in model.vocab and "i_{}".format(item) in model.vocab]
    test_vec, ground_truth = zip(*test_data)
    label_encoder = preprocessing.LabelEncoder()
    test_vec = np.array(test_vec)
    ground_truth = np.array(ground_truth)
    ground_truth = label_encoder.fit_transform(ground_truth)

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
    return (train_vectors,train_labels,test_vectors,test_labels,label_encoder)


def get_multi_score(clf,x,y,label):
    pred = clf.predict(x)
    acc = accuracy_score(y,pred)
    mse = mean_squared_error(label.inverse_transform(y),label.inverse_transform(pred))

    return (acc,mse)



def k_sim(model, db,norm=False):

    train_vectors,train_labels,test_vectors,test_labels,label_encoder =  get_dataset(model,db,norm)


    for c in [1,2,3,4,5]:
        clf = linear_model.LogisticRegression(C=c,dual=False,multi_class="multinomial",solver="lbfgs")
        clf.fit(train_vectors,train_labels)

        print("TRAIN: accuracy: {}, MSE: {}".format(*get_multi_score(clf,train_vectors,train_labels,label_encoder)))
        print("TEST: accuracy: {}, MSE: {}".format(*get_multi_score(clf,test_vectors,test_labels,label_encoder)))






parser = argparse.ArgumentParser()

parser.add_argument("model", type=str)
parser.add_argument("db", type=str)
parser.add_argument("--norm",dest="norm",action="store_true")

args = parser.parse_args()
db = args.db
norm = args.norm
model = Doc2Vec.load_word2vec_format(args.model, binary=True,norm_only=False)
k_sim(model, db,norm)
