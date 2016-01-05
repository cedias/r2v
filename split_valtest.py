import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from gensim import matutils
from random import shuffle
import pickle
from sklearn.preprocessing import normalize

def getAllReviews(db,test=False):
    con = sqlite3.connect(db)
    c = con.cursor()
    if test:
        c.execute("SELECT item,user,rating FROM reviews WHERE test")
    else:
        c.execute("SELECT item,user,rating FROM reviews WHERE not test")
    return c.fetchall()


def split(db):

    test_data = [(item, user, rating) for item, user, rating in getAllReviews(db, test=True)]
    shuffle(test_data)
    val = test_data[len(test_data)//2:]
    test = test_data[:len(test_data)//2]
    print("#Val: {}  #Test: {}".format(len(val),len(test)))
    return (val,test)


parser = argparse.ArgumentParser()

parser.add_argument("db", type=str)
parser.add_argument('output',type=str)


args = parser.parse_args()
db = args.db
val,test = split(db)
result = (val,test)
pickle.dump(result,open(args.output,"wb"))
