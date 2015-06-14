import sqlite3
import argparse
import numpy as np
import itertools
from random import shuffle,choice, randint

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


def k_sim(db,neigh="user",n=0):

    if neigh not in {"user","item"}:
        print("only {} as similarity".format(["user","item"]))

    print("prepping data")
    test_data = [(item, user, rating) for item, user, rating in getAllReviews(db, test=True)]
    shuffle(test_data)
    print("test data ready")

    cpt_test = 0
    cpt_skipped = 0

    oracle_r1 = 0
    oracle_r2 = 0
    oracle_r3 = 0

    random_r1 = 0
    random_r2 = 0
    random_r3 = 0

    for item, user, _ in test_data:

        if cpt_test >= len(test_data)/2: # we only evaluate on random 50%
            break

        if neigh == "user":
            list_text = [stext for _,_,stext in getItemReviews(item, db)]

        elif neigh == "item":
            list_text = [stext for _,_,stext in getUserReviews(user, db)]
        else:
            raise Exception("Neigh not item nor user")

        if len(list_text) == 0:
            cpt_skipped +=1
            continue

        rtext = getReviewText(db,user,item)
        rtext = rtext.replace("."," ").lower().split(" ")

        if len(rtext) < 3: #division by zero, duh !
            cpt_skipped +=1
            continue


        if n <= 0:

            list_text = [t.replace("."," ").lower().split(" ") for t in list_text]
            rouges = [rouge_1_2_3_metric(rtext,t) for t in list_text ]

            rand_choice = choice(rouges)
            best_choice = np.max(rouges,axis=0)

            oracle_r1 += best_choice[0]
            oracle_r2 += best_choice[1]
            oracle_r3 += best_choice[2]

            random_r1 += rand_choice[0]
            random_r2 += rand_choice[1]
            random_r3 += rand_choice[2]
        else:

            sents = list(itertools.chain.from_iterable([x.split(".") for x in list_text]))
            sents_ch = [x for x in sents if len(x)>2]
            sents_rouges = [x.replace("."," ").lower().split(" ") for x in sents_ch]

            if(len(sents_ch) < n):
                cpt_skipped += 1
                continue



            rouges = [rouge_1_2_3_metric(rtext,t) for t in sents_rouges] #metric rouge pour toutes les phrases

            if len(rouges) < n: #pas assez de phrases
                cpt_skipped += 1
                continue

            best_choices =  np.argsort(rouges,axis=0)[::-1].T


            best_choices_r1 =  list(itertools.chain.from_iterable([ sents_rouges[i] for i in best_choices[0][:n]]))
            best_choices_r2 =  list(itertools.chain.from_iterable([ sents_rouges[i] for i in best_choices[1][:n]]))
            best_choices_r3 =  list(itertools.chain.from_iterable([ sents_rouges[i] for i in best_choices[2][:n]]))

            words_real_n1 = find_ngrams(rtext,1)
            words_pred_n1 = find_ngrams(best_choices_r1,1)
            words_real_n2 = find_ngrams(rtext,2)
            words_pred_n2 = find_ngrams(best_choices_r2,2)
            words_real_n3 = find_ngrams(rtext,3)
            words_pred_n3 = find_ngrams(best_choices_r3,3)

            oracle_r1 += len(words_real_n1.intersection(words_pred_n1))/(len(words_real_n1)+0.0)
            oracle_r2 += len(words_real_n2.intersection(words_pred_n2))/(len(words_real_n2)+0.0)
            oracle_r3 += len(words_real_n3.intersection(words_pred_n3))/(len(words_real_n3)+0.0)

            for i in range(0,n):
                rand_i = randint(0,len(rouges)-1)
                rand_choice = rouges[rand_i]

                random_r1 += rand_choice[0]
                random_r2 += rand_choice[1]
                random_r3 += rand_choice[2]

                del(rouges[rand_i])





        cpt_test += 1
        if cpt_test % 100 == 0:
            print("at {} -- R1:[{},{}], R2:[{},{}], R3:[{},{}] (skipped {})".format(cpt_test, random_r1/(cpt_test+0.0),oracle_r1/(cpt_test+0.0),random_r2/(cpt_test+0.0),oracle_r2/(cpt_test+0.0),random_r3/(cpt_test+0.0),oracle_r3/(cpt_test+0.0),cpt_skipped))

    print("at {} --  R1:[{},{}], R2:[{},{}], R3:[{},{}] (skipped {})".format(cpt_test, random_r1/(cpt_test+0.0),oracle_r1/(cpt_test+0.0),random_r2/(cpt_test+0.0),oracle_r2/(cpt_test+0.0),random_r3/(cpt_test+0.0),oracle_r3/(cpt_test+0.0),cpt_skipped))

parser = argparse.ArgumentParser()

parser.add_argument("--neigh",default="item", type=str)
parser.add_argument("--n",default=0, type=int)
parser.add_argument("db", type=str)

args = parser.parse_args()
db = args.db
k_sim(db,neigh=args.neigh,n=args.n)
