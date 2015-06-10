import numpy as np



class PredictionModelIterator(object):

    def __init__(self,model,db,predmod=None,metrics=None):
        self.db = db
        self.model = model
        self.predmod = predmod
        self.metrics = metrics

    def fit_all(self):
        for mod in self.predmod:
            mod.fit(self.model,self.db)

    def evaluate_all(self,output=None):
        for mod in self.predmod:
            print(mod.pretty_print())
            if output is not None:
                output.write(mod.pretty_print()+"\n")
            for metric in self.metrics:
                m = metric(mod,self.db)
                m.evaluate()
                print(m.pretty_print())
                if output is not None:
                    output.write(m.pretty_print()+"\n")



class Metric(object):
    def __init__(self, pred_model, db):
        self.pred_model = pred_model
        self.db = db
        self.value = 0

    def evaluate(self):
        pass
    def pretty_print(self):
        return "Value for {} is {}".format(self.__class__.__name__,self.value)

class MSE(Metric):
    def evaluate(self):
        diff = [self.pred_model.predict(item,user)-rating for item,user,rating in self.db.getAllReviews(test=True) if self.pred_model.predict(item,user) != None]
        sqDiff = [ x*x for x in diff]
        self.value = np.mean(sqDiff)


class PredictionModel(object):
    def fit(self,model,db):
        pass

    def predict(self, user, item):
        pass

    def pretty_print(self):
        return "\n{} Prediction Model".format(self.__class__.__name__)


class OverallBiasDB(PredictionModel):

    def fit(self,model,db):
        self.db_avg = db.getOverallBias()[0]

    def predict(self, user, item):
        return self.db_avg

    def pretty_print(self):
        return "\nOverall DB Bias: {}".format(self.db_avg)


class OverallBiasSpace(PredictionModel):
    def fit(self,model,db):
        self.space_avg = np.mean([model.most_similar_rating(vect=((model["u_"+str(u)]+model["i_"+str(i)])/2)) for u, i, r in db.getAllReviews(test=False) if "u_"+str(u) in model.vocab and "i_"+str(i) in model.vocab])

    def predict(self, user, item):
        return  self.space_avg

    def pretty_print(self):
        return "\nOverall Space Bias: {}".format(self.space_avg)


class ItemBiasDB(PredictionModel):
    def fit(self,model,db):
        self.db = db

    def predict(self, user, item):
        return self.db.getItemBias(item)


class UserBiasDB(PredictionModel):
    def fit(self,model,db):
        self.db = db

    def predict(self, user, item):
        return self.db.getUserBias(user)


class ItemBiasSpace(PredictionModel):
    def fit(self,model,db):
        self.model = model

    def predict(self, user, item):
        if "i_"+str(item) in self.model.vocab:
            return self.model.most_similar_rating(self.model["i_"+str(item)])
        else:
            return None


class UserBiasSpace(PredictionModel):
    def fit(self,model,db):
        self.model = model

    def predict(self, user, item):
        if "u_"+str(user) in self.model.vocab:
            return self.model.most_similar_rating(self.model["u_"+str(user)])
        else:
            return None


class ClassicDB(PredictionModel):
    def fit(self,model,db):
        self.db = db
        self.db_avg = db.getOverallBias()[0]

    def predict(self, user, item):
        ub = self.db.getUserBias(user)
        ib = self.db.getItemBias(item)

        if ub is None:
            ub = 0
        if ib is None:
            ib = 0

        return (self.db_avg + (ib-self.db_avg) + (ub-self.db_avg))


class ClassicSpace(PredictionModel):
    def fit(self,model,db):
        self.space_avg = np.mean([model.most_similar_rating(vect=((model["u_"+str(u)]+model["i_"+str(i)])/2)) for u, i, r in db.getAllReviews(test=False) if "u_"+str(u) in model.vocab and "i_"+str(i) in model.vocab])
        self.model = model

    def predict(self, user, item):
        if "u_"+str(user) not in self.model.vocab or "i_"+str(item) not in self.model.vocab:
            return None
        else:
            return (self.space_avg + self.model.most_similar_rating(self.model["u_"+str(user)]) + self.model.most_similar_rating(self.model["i_"+str(item)]) )/3.0


class ClassicMean(PredictionModel):
    def fit(self,model,db):
        self.space_avg = np.mean([model.most_similar_rating(vect=((model["u_"+str(u)]+model["i_"+str(i)])/2)) for u, i, r in db.getAllReviews(test=False) if "u_"+str(u) in model.vocab and "i_"+str(i) in model.vocab])
        self.model = model
        self.db = db
        self.db_avg = db.getOverallBias()[0]


    def predict(self, user, item):
        ub = self.db.getUserBias(user)
        ib = self.db.getItemBias(item)

        if ub is None:
            ub = self.db_avg
        if ib is None:
            ib = self.db_avg

        if "u_"+str(user) not in self.model.vocab or  "i_"+str(item) not in self.model.vocab:
            return None

        a =  (self.db_avg + ub + ib)/3.0
        b =  (self.space_avg + self.model.most_similar_rating(self.model["u_"+str(user)]) + self.model.most_similar_rating(self.model["i_"+str(item)]) )/3.0
        return  (a + b) /2.0


class CollabFiltering(PredictionModel):

    def __init__(self,k,verbose=True):
        self.k = k
        self.skipped = 0
        self.cpt_test = 0
        self.verbose = verbose
        self._cache = None
        self._usercache = None

    def fit(self,model,db):
        print("fitting collaborative filtering")
        self.db = db
        self.model = model
        i=0

        self.model._cache = self.model.model.syn0norm[self.model.user_indexs] #Hack

        print("Collab filtering fitted")


    def predict(self, user, item):
        if self.verbose and (self.cpt_test+self.skipped)% 100 == 0:
            print("Saw {} tests, skipped {} - Total {}/426305 ".format(self.cpt_test,self.skipped, self.cpt_test+self.skipped))
        if "u_{}".format(user) not in self.model.vocab:
            self.skipped += 1
            return None
        else:
            sim_users = {suser:srating for suser,srating,_ in self.db.getItemReviews(item, test=False)} #note déja données
            user_sims = self._cache_sim(user) #similarité user/users
            cpt = 0
            sum_r=0
            sum_sim=0

            for sim_user,sim_value in user_sims:
                if sim_user in sim_users:
                    sum_r += sim_users[sim_user]*sim_value
                    sum_sim += sim_value
                    cpt +=1

                if cpt >= self.k:
                    self.cpt_test += 1
                    return sum_r/(sum_sim+0.0)
            self.cpt_test +=1

            if sum_sim == 0:
                self.skipped += 1
                return None

            return sum_r/(sum_sim+0.0)


    def _cache_sim(self,user):
        if user == self._usercache:
            return self._cache
        else:
            self._usercache = user
            self._cache = self.model.most_similar_cache(self.model["u_{}".format(user)])
            return self._cache
