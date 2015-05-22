from VectReco.Database import Database
from VectReco.R2VModel import R2VModel
from VectReco.PredictionModel import PredictionModelIterator,MSE,OverallBiasDB,OverallBiasSpace,ItemBiasDB,ItemBiasSpace,UserBiasDB,UserBiasSpace,ClassicDB,ClassicSpace,ClassicMean

def run():
    mod = R2VModel.from_w2v_text("/local/dias/rb_10000_i10_s200_n5_08")
    db = Database("/local/dias/db/ratebeer.db")

    predmod = [OverallBiasDB(),OverallBiasSpace(),ItemBiasDB(),ItemBiasSpace(),UserBiasDB(),UserBiasSpace(),ClassicDB(),ClassicSpace(),ClassicMean()]
    metrics = [MSE]

    pmi = PredictionModelIterator(mod,db,predmod=predmod,metrics=metrics)
    pmi.fit_all()
    pmi.evaluate_all()


if __name__ == '__main__':
    run()

