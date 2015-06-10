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
        c.execute("SELECT item,user,rating FROM reviews WHERE test ORDER BY user")
    else:
        c.execute("SELECT item,user,rating FROM reviews WHERE not test ORDER BY user")
    return c.fetchall()

def getReviewText(db,user,item):
    con = sqlite3.connect(db)
    c = con.cursor()
    c.execute("SELECT review FROM reviews WHERE user = {} and item = {}".format(user,item))
    return c.fetchone()[0]

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



def find_ngrams(input_list, n):
  return {" ".join(p) for p in zip(*[input_list[i:] for i in range(n)])}


def rouge_1_2_3_metric(words_real,words_pred):
    words_real_n1 = find_ngrams(words_real,1)
    words_pred_n1 = find_ngrams(words_pred,1)

    words_real_n2 = find_ngrams(words_real,2)
    words_pred_n2 = find_ngrams(words_pred,2)

    words_real_n3 = find_ngrams(words_real,3)
    words_pred_n3 = find_ngrams(words_pred,3)

    r1 = len(words_real_n1.intersection(words_pred_n1))/(len(words_real_n1)+0.0)
    r2 = len(words_real_n2.intersection(words_pred_n2))/(len(words_real_n2)+0.0)
    r3 = len(words_real_n3.intersection(words_pred_n3))/(len(words_real_n3)+0.0)

    return (r1,r2,r3)


def k_sim(model, db,neigh="user"):

    if neigh not in {"user","item"}:
        print("only {} as similarity".format(["user","item"]))

    print("prepping data")
    test_data = [(item, user, rating) for item, user, rating in getAllReviews(db, test=True)]
    shuffle(test_data)
    print("test data ready")


    cpt_test = 0
    cpt_skipped = 0
    r1s,r2s,r3s = 0,0,0
    oracle = 0

    for item, user, _ in test_data:

        if cpt_test >= len(test_data)/2: # we only evaluate on random 50%
            break

        if "u_{}".format(user) not in model.vocab or "i_{}".format(item) not in model.vocab: #skip not in vocab
            cpt_skipped += 1
            continue

        if neigh=="user":
            vect = model["u_{}".format(user)]
        elif neigh=="item":
            vect = model["i_{}".format(item)]
        else:
            raise Exception("Neigh not item nor user")

        vect = matutils.unitvec(vect)

        if neigh == "user":
            list_sims = [(stext,np.dot(matutils.unitvec(model["u_{}".format(suser)]),vect)) for suser,_,stext in getItemReviews(item, db) if "u_{}".format(suser) in model.vocab]

        elif neigh == "item":
            list_sims = [(stext,np.dot(matutils.unitvec(model["i_{}".format(sitem)]),vect)) for sitem,_,stext in getUserReviews(user, db) if "i_{}".format(sitem) in model.vocab]
        else:
            raise Exception("Neigh not item nor user")

        if len(list_sims) == 0:
            cpt_skipped +=1
            continue

        sim_text, sim_sim = zip(*list_sims)
        maxim_i = np.argmax(sim_sim)

        rtext = getReviewText(db,user,item)
        ptext = sim_text[maxim_i]

        rtext = rtext.replace("."," ").lower().split(" ")
        ptext = ptext.replace("."," ").lower().split(" ")

        if len(rtext) < 3:
            cpt_skipped += 1
            continue

        r1,r2,r3 = rouge_1_2_3_metric(rtext,ptext)
        r1s += r1
        r2s += r2
        r3s += r3

        cpt_test += 1
        if cpt_test % 100 == 0:
            print("Tests is {}, rouge1 is {}, rouge2 is {}, rouge3 is {} - {} test cases where skipped".format(cpt_test,r1s/(cpt_test+0.0),r2s/(cpt_test+0.0),r3s/(cpt_test+0.0),cpt_skipped))

    print("Final for {} tests is {}, rouge1 is {}, rouge2 is {}, rouge3 is {}  - {} test cases where skipped".format(cpt_test,r1s/(cpt_test+0.0),r2s/(cpt_test+0.0),r3s/(cpt_test+0.0),cpt_skipped))


parser = argparse.ArgumentParser()

parser.add_argument("--neigh",default="item", type=str)
parser.add_argument("model", type=str)
parser.add_argument("db", type=str)

args = parser.parse_args()
db = args.db
model = Doc2Vec.load_word2vec_format(args.model, binary=True,norm_only=False)
k_sim(model,db,neigh=args.neigh)
