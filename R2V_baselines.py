from VectReco.Database import Database
from VectReco.R2VModel import R2VModel
from VectReco.PredictionModel import PredictionModelIterator,MSE,OverallBiasDB,OverallBiasSpace,ItemBiasDB,ItemBiasSpace,UserBiasDB,UserBiasSpace,ClassicDB,ClassicSpace,ClassicMean,CollabFiltering
import  argparse


def run(args):
    print("Loading datas")
    mod = R2VModel.from_w2v_text(args.model)
    db = Database(args.db)
    print("Loading over")
    predmod = [OverallBiasDB(),OverallBiasSpace(),ItemBiasDB(),ItemBiasSpace(),UserBiasDB(),UserBiasSpace(),ClassicDB(),ClassicSpace(),ClassicMean(),CollabFiltering(args.nb)]
    metrics = [MSE]
    output_file = None
    if args.output is not None:
        output_file = open(args.output,"w")

    pmi = PredictionModelIterator(mod,db,predmod=predmod,metrics=metrics)
    pmi.fit_all()
    pmi.evaluate_all(output=output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--output",default=None, type=str)
    parser.add_argument("--nb",default=100, type=int)
    args = parser.parse_args()
    run(args)

