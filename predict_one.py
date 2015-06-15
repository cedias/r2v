import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from gensim import matutils
import itertools
from sklearn.preprocessing import normalize


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


def getReview(user,item, db):
    con = sqlite3.connect(db)
    c = con.cursor()
    c.execute("SELECT rating,review FROM reviews WHERE user = {} and item = {}".format(user,item))
    return c.fetchone()

def getAllReviews(db,test=False):
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

    print(words_real_n1.intersection(words_pred_n1))

    return (r1,r2,r3)

def predict_text(model,vect,texts,num_sent):
    sents = list(itertools.chain.from_iterable([x.split(".") for x in texts]))
    sents = [x for x in sents if len(x)>2]

    if len(sents) < num_sent:
        print("not enough sentences")
        return None

    sentences = np.zeros((len(sents),model.layer1_size))

    for i,sent in enumerate(sents):
        for word in sent.split():
            if word in model.vocab:
                sentences[i] += model[word]

    sentences = normalize(sentences,axis=1,norm="l1")
    sim = np.dot(sentences,vect)
    order = np.argsort(sim)[::-1]

    text = ""
    sents = [sents[a] for a in order[:num_sent]]
    return ".".join(sents)

def k_sim(model, db,user,item,k=None,neigh="user",mean_norm=False,n=0):

    if neigh not in {"user","item"}:
        print("only {} as similarity".format(["user","item"]))
        return None

    original = getReview(user,item,db)
    if original is None:
        print("no review from user {} on item {}".format(user,item))
        return None

    original_rating, original_review  = original

    if mean_norm:
        print("prepping data")
        if neigh == "user":
            u_bias = {user:bias for user,bias in getUsersBias(db)}
        elif neigh == "item":
            i_bias = {item:bias for item,bias in getItemsBias(db)}
        print("prepping ok")

    if "u_{}".format(user) not in model.vocab and neigh == "user":
        print("{} not in model".format(user))
        return None

    if "i_{}".format(item) not in model.vocab and neigh == "item":
        print("{} not in model".format(item))
        return None

    if neigh=="user":
        vect = model["u_{}".format(user)]
    elif neigh=="item":
        vect = model["i_{}".format(item)]
    else:
        raise Exception("Neigh not item nor user")

    vect = matutils.unitvec(vect)

    if neigh == "user":
        if mean_norm:
            list_sims = [(suser,srating-u_bias[suser],np.dot(matutils.unitvec(model["u_{}".format(suser)]),vect),stext) for suser,srating,stext in getItemReviews(item, db) if "u_{}".format(suser) in model.vocab]
        else:
            list_sims = [(suser,srating,np.dot(matutils.unitvec(model["u_{}".format(suser)]),vect),stext) for suser,srating,stext in getItemReviews(item, db) if "u_{}".format(suser) in model.vocab]

    elif neigh == "item":
        if mean_norm:
            list_sims = [(sitem,srating-i_bias[sitem],np.dot(matutils.unitvec(model["i_{}".format(sitem)]),vect),stext) for sitem,srating,stext in getUserReviews(user, db) if "i_{}".format(sitem) in model.vocab]
        else:
            list_sims = [(sitem,srating,np.dot(matutils.unitvec(model["i_{}".format(sitem)]),vect),stext) for sitem,srating,stext in getUserReviews(user, db) if "i_{}".format(sitem) in model.vocab]


    if len(list_sims) == 0:
        print("No similar reviews...")
        return None

    sim_users,sim_rating,sim_sim,sim_text = zip(*list_sims)

    sim_sim = [(x+1)/(2+0.0) for x in sim_sim] # make sim between [0,1]
    order = np.argsort(sim_sim)[::-1]


    if k is not None:
        order = order[:k]


    sum_rs = sum([sim_rating[x]*sim_sim[x] for x in order])
    sum_sim = sum([sim_sim[x] for x in order])

    predicted_rating = sum_rs/(sum_sim+0.0)

    if n==0:
        predicted_text = sim_text[order[0]]
    else:
        sim_text = list(sim_text)
        predicted_text = predict_text(model,vect,sim_text,n)

        if predicted_text == None:
            print("No predicted text")
            return None

    if mean_norm:
        if neigh == "user":
            predicted_rating +=  u_bias[user]
        elif neigh == "item":
            predicted_rating +=  i_bias[item]


    return (original_rating,original_review,predicted_rating,predicted_text)


parser = argparse.ArgumentParser()

parser.add_argument("--n",default=1, type=int)
parser.add_argument("--neigh",default="item",type=str)
parser.add_argument('--mean_center', dest='mean_center', action='store_true')
parser.add_argument("model", type=str)
parser.add_argument("db", type=str)
parser.add_argument("user",type=int)
parser.add_argument("item",type=int)
args = parser.parse_args()

db = args.db
model = Doc2Vec.load_word2vec_format(args.model, binary=True,norm_only=False)

res = k_sim(model,db,args.user,args.item,neigh=args.neigh,n=args.n,mean_norm=args.mean_center)
res_full = k_sim(model,db,args.user,args.item,neigh=args.neigh,n=0,mean_norm=args.mean_center)

if res is None:
    print("Can't predict Anything")
else:
    ora,ore,pra,pre = res
    _,_,_,full_rev = res_full

    print("Original rating: {} , Original review: {} ".format(ora,ore))
    print("Predicted rating is {}, predicted review is {}".format(pra,pre))
    print("Predicted rating is {}, predicted full_review is {}".format(pra,full_rev))
