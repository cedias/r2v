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

    def evaluate_all(self):
        for mod in self.predmod:
            mod.pretty_print()
            for metric in self.metrics:
                m = metric(mod,self.db)
                m.evaluate()
                m.pretty_print()



class Metric(object):
    def __init__(self, pred_model, db):
        self.pred_model = pred_model
        self.db = db
        self.value = 0

    def evaluate(self):
        pass
    def pretty_print(self):
        print("Value for {} is {}".format(self.__class__.__name__,self.value))

class MSE(Metric):
    def evaluate(self):
        diff = [self.pred_model.predict(item,user)-rating for item,user,rating in self.db.getAllReviews(test=True)]
        sqDiff = [ x*x for x in diff]
        self.value = np.mean(sqDiff)


class PredictionModel(object):
    def fit(self,model,db):
        pass

    def predict(self, user, item):
        pass

    def pretty_print(self):
        pass


class OverallBiasDB(PredictionModel):

    def fit(self,model,db):
        self.db_avg = db.getOverallBias()[0]

    def predict(self, user, item):
        return self.db_avg

    def pretty_print(self):
        print("\nOverall DB Bias: {}".format(self.db_avg))


class OverallBiasSpace(PredictionModel):
    def fit(self,model,db):
        self.space_avg = np.mean([model.most_similar_rating(vect=((model["u_"+str(u)]+model["i_"+str(i)])/2)) for u, i, r in db.getAllReviews(test=False) if "u_"+str(u) in model.vocab and "i_"+str(i) in model.vocab])

    def predict(self, user, item):
        return  self.space_avg

    def pretty_print(self):
        print("\nOverall Space Bias: {}".format(self.space_avg))


class ItemBias(PredictionModel):
    def fit(self,model,db):
        pass
    def pretty_print(self):
        pass

class UserBias(PredictionModel):
    def fit(self,model,db):
        pass
    def pretty_print(self):
        pass

class Classic(PredictionModel):
    def fit(self,model,db):
        pass
    def pretty_print(self):
        pass

class CollabFiltering(PredictionModel):
    def fit(self,model,db):
        pass
    def pretty_print(self):
        pass