import sqlite3
import argparse
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from gensim import matutils
from random import shuffle
import pickle
from sklearn.preprocessing import normalize
from io import StringIO

def getItemReviews(item, db):
	c = db.cursor()
	c.execute("SELECT user,rating,review FROM reviews WHERE item = {} and not test".format(item))
	return c.fetchall()

def getUserReviews(user, db):

	c = db.cursor()
	c.execute("SELECT item,rating,review FROM reviews WHERE user = {} and not test".format(user))
	return c.fetchall()

def getAllReviews(db,test=False):
	c = db.cursor()
	if test:
		c.execute("SELECT item,user,rating FROM reviews WHERE test")
	else:
		c.execute("SELECT item,user,rating FROM reviews WHERE not test")
	return c.fetchall()

def getUsersBias(db):
	c = db.cursor()
	c.execute("SELECT user,avg(rating) as bias FROM reviews WHERE not test group by user")
	return c.fetchall()

def getItemsBias(db):
	c = db.cursor()
	c.execute("SELECT item, avg(rating) as bias FROM reviews WHERE not test group by item")
	return c.fetchall()

def getFullBias(db):
    c = db.cursor()
    c.execute("SELECT avg(rating) as bias FROM reviews")
    return c.fetchone()[0]



def init_sqlite_db(db):
	# Read database to tempfile
	con = sqlite3.connect(db)
	tempfile = StringIO()
	for line in con.iterdump():
		tempfile.write('%s\n' % line)
	con.close()
	tempfile.seek(0)
	# Create a database in memory and import from tempfile
	db = sqlite3.connect(":memory:")
	db.cursor().executescript(tempfile.read())
	db.commit()
	return db


def k_sim(model, db,data,k=None,neigh="user",mean_norm=False):

	if neigh not in {"user","item","sum","CSu","CSi"}:
		print("only {} as similarity".format(["user","item","sum","CSu","CSi"]))

	print("prepping data")
	if mean_norm:
		if neigh == "user":
			u_bias = {user:bias for user,bias in getUsersBias(db)}
		elif neigh == "item":
			i_bias = {item:bias for item,bias in getItemsBias(db)}
	if neigh == "CSu" or neigh == "CSi":
		full_mean = getFullBias(db)

	test_data = data
	
	print("test data ready")

	cpt_test = 0
	cpt_skipped = 0
	tot_err = np.zeros(k)

	for item, user, rating in test_data:

		if ("u_{}".format(user) not in model.vocab and (neigh=="user" or neigh=="CSu")) or ("i_{}".format(item) not in model.vocab and (neigh=="item" or neigh=="CSi")) or (("u_{}".format(user) not in model.vocab or "i_{}".format(item) not in model.vocab) and neigh=="sum") : #skip not in vocab
			cpt_skipped += 1
			continue

		if neigh=="user" or neigh == "CSu":
			vect = model["u_{}".format(user)]
		elif neigh=="item" or neigh == "CSi":
			vect = model["i_{}".format(item)]
		elif neigh=="sum":
			vect = model["i_{}".format(item)] + model["u_{}".format(user)]
		else:
			raise Exception("Neigh not item nor user")

		vect = matutils.unitvec(vect)

		if neigh == "user" or neigh == "CS":
			if mean_norm:
				list_sims = [(suser,srating-u_bias[suser],model["u_{}".format(suser)]) for suser,srating,_ in getItemReviews(item, db) if "u_{}".format(suser) in model.vocab]
			else:
				list_sims = [(suser,srating,model["u_{}".format(suser)]) for suser,srating,_ in getItemReviews(item, db) if "u_{}".format(suser) in model.vocab]

		elif neigh == "item":
			if mean_norm:
				list_sims = [(sitem,srating-i_bias[sitem],model["i_{}".format(sitem)]) for sitem,srating,_ in getUserReviews(user, db) if "i_{}".format(sitem) in model.vocab]
			else:
				list_sims = [(sitem,srating,model["i_{}".format(sitem)]) for sitem,srating,_ in getUserReviews(user, db) if "i_{}".format(sitem) in model.vocab]

		elif neigh == "sum":
				list_sims = [(suser,srating,model["u_{}".format(suser)]+model["i_{}".format(item)]) for suser,srating,_ in getItemReviews(item, db) if "u_{}".format(suser) in model.vocab]
				list_sims += [(sitem,srating,model["i_{}".format(sitem)]+model["u_{}".format(user)]) for sitem,srating,_ in getUserReviews(user, db) if "i_{}".format(sitem) in model.vocab]


		if len(list_sims) == 0:
			cpt_skipped += 1
			continue

		sim_users,sim_rating,sim_sim = zip(*list_sims)
		sim_rating = np.array(sim_rating)
		sim_sim = np.array(sim_sim)

		sim_sim = normalize(sim_sim, copy=False)
		sim_sim = np.dot(vect,sim_sim.T)

		sim_sim += 1
		sim_sim /= 2.0
		order = np.argsort(sim_sim)[::-1]

		if len(order) == 0:
			cpt_skipped +=1
			continue
		
	  

		order = np.array(order)
		sim_sim = sim_sim[order]
		sim_rating = sim_rating[order]

		sim_sim = sim_sim[:k]
		sim_rating = sim_rating[:k]
		
		if neigh == "CSu" or neigh == "CSi":
			sim_rating -= full_mean

		pond = sim_rating * sim_sim

		sum_rs = np.cumsum(pond)
		sum_sim = np.cumsum(sim_sim)
	   

		predicted = sum_rs/(sum_sim+0.0)


		if mean_norm:
			if neigh == "user":
				predicted +=  u_bias[user]
			elif neigh == "item":
				predicted +=  i_bias[item]

		if neigh == "CSu" or neigh == "CSi":
			predicted += full_mean



		err = (rating - predicted) ** 2

		if len(err != k):
			oldlen = len(err)
			err.resize(k,refcheck=False)
			err[oldlen:k] = err[oldlen-1]

		tot_err += err

		

		cpt_test += 1
		
		if cpt_test % 1000 == 0:
			print("MSE at {} tests is {} - {} test cases where skipped".format(cpt_test,tot_err/(cpt_test+0.0),cpt_skipped))

	print("Final MSE for {} tests is {} - {} test cases where skipped".format(cpt_test,tot_err/(cpt_test+0.0),cpt_skipped))
	return tot_err/(cpt_test+0.0)


parser = argparse.ArgumentParser()

parser.add_argument("--k",default=5, type=int)
parser.add_argument("--neigh",default="item", type=str)
parser.add_argument('--mean_center', dest='mean_center', action='store_true')
parser.add_argument('--output', default="predicted_ratings.pkl",type=str)
parser.add_argument('--validation',  dest='validation', action='store_true')
parser.add_argument('--ram',  dest='ram', action='store_true')

parser.add_argument("model", type=str)
parser.add_argument("db", type=str)
parser.add_argument("data", type=str)


args = parser.parse_args()

if not args.ram:
	db = sqlite3.connect(args.db)
else:
	db = init_sqlite_db(args.db)

val,test = pickle.load(open(args.data,"rb"))


if args.validation:
	data = val
else:
	data = test

model = Doc2Vec.load_word2vec_format(args.model, binary=True,norm_only=False)
err = k_sim(model,db,data,k=args.k,neigh=args.neigh,mean_norm=args.mean_center)

result = {
	"neigh":args.neigh,
	"mean_center":args.mean_center,
	"model":args.model,
	"db":args.db,
	"validation":args.validation,
	"error":err
}

pickle.dump(result,open(args.output,"wb"))
