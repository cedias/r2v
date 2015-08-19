import argparse
from gensim.models import Doc2Vec


parser = argparse.ArgumentParser()

parser.add_argument("type",type=str)
parser.add_argument("id", type=str)
parser.add_argument("db", type=str)


args = parser.parse_args()
