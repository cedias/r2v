import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from random import shuffle,randint
from sklearn import preprocessing,cross_validation,linear_model
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, mean_squared_error
from gensim import matutils
import graphlab as gl
from graphlab import SFrame



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


def get_multi_score(clf,x,y,vtarget):
    pred = clf.predict(x)

    if not vtarget:
        mse = mean_squared_error(y,pred)

    else:

        rating_indexs = [(matutils.unitvec(model[word]), float(word.split("_")[1])) for word in model.index2word if len(word) > 2 and word[0] == "r" and word[1] == "_"]
        r_vec, ratings = zip(*rating_indexs)
        ratings = np.array(ratings)
        r_vec = np.array(r_vec).T
        dp = np.dot(pred,r_vec)
        dp2 = np.dot(y,r_vec)
        print(np.argmax(dp, axis=1))
        err = (ratings[np.argmax(dp, axis=1)] - ratings[np.argmax(dp2, axis=1)]) ** 2
        mse = np.mean(err)

    return mse



def k_sim(model, db,norm=False,conc=False,vtarget=False):

    train_vectors,train_labels,test_vectors,test_labels =  get_dataset(model,db,norm,conc,vtarget)

    train = {}
    train["data"] = train_vectors
    train["labels"]= train_labels
    train = SFrame(train)
    print(train)

    test = {}
    test["data"] = test_vectors
    test["labels"]= test_labels
    test = SFrame(test)
    print(test)


    # params = dict([('l2_penalty', [0,0.00001,0.0001,0.001,0.01,0.1,1]),
    #                ('l1_penalty', [0,0.00001,0.0001,0.001,0.01,0.1,1]),
    #                ('target','labels'),
    #                ('feature_rescaling',[True,False])
    #                ])
    #
    # job = gl.random_search.create((train, test),
    #                             gl.linear_regression.create,
    #                             params,return_model=False,max_models=25)
    #
    # results = job.get_results()
    # print(results)
    # results.save('random_search_reg', format='csv')


    params = dict([('max_iterations', [10,20,30]),
                   ('target','labels')])



    job = gl.random_search.create((train, test),
                                gl.neuralnet_classifier.create,
                                params,return_model=False,max_models=25)

    results = job.get_results()
    print(results)
    results.save('random_search_nn', format='csv')




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
