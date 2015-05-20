from VectReco.Database import Database
from VectReco.R2VModel import R2VModel
from VectReco.DatasetIterator import AmazonIterator,BeeradvocateIterator,RatebeerIterator
import argparse

def run(args):
    types = {"amazon":AmazonIterator,"ratebeer":RatebeerIterator,"beeradvocate":BeeradvocateIterator}

    if args.type not in types:
        print("data type not supported, supported types are {} ".format(types.keys()))

    data_iterator = types[args.type](args.data)
    db = Database.build(args.output,data_iterator.iterate())


parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("type",type=str)
parser.add_argument("output", type=str)
parser.set_defaults(train_words=True)
args = parser.parse_args()


if __name__ == '__main__':
    run(args)

