from VectReco.Database import Database
from VectReco.DatasetIterator import AmazonIterator,BeeradvocateIterator,RatebeerIterator, YelpIterator
import argparse

def run(args):
    types = {"amazon":AmazonIterator,"ratebeer":RatebeerIterator,"beeradvocate":BeeradvocateIterator,"yelp":YelpIterator}

    if args.type not in types:
        print("data type not supported, supported types are {} ".format(types.keys()))

    print(args.zipped)

    data_iterator = types[args.type](args.data,zipped=args.zipped,encoding=args.encoding)
    db = Database.build(args.output,data_iterator.iterate())


parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("type",type=str)
parser.add_argument("--encoding",default="utf-8", type=str)
parser.add_argument('--gz', dest='zipped', action='store_true')
parser.add_argument("output", type=str)
parser.set_defaults(train_words=True)
args = parser.parse_args()


if __name__ == '__main__':
    run(args)

