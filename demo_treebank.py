#coding: utf-8

import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--threads",default=5, type=int)
args = parser.parse_args()

subprocess.call(["python3.4 treebank_to_R2V.py --output treebank.d2v Data/stanfordSentimentTreebank"],shell=True)
subprocess.call(["./d2v/d2v -train treebank.d2v -output treebank_d2v.bin -binary 1 -hs 0 -window 10 -sample 0 -min-count 0 -negative 15 -sentence-vectors 1 -cbow 0 -iter 10 -threads {}".format(args.threads)],shell=True)
subprocess.call(["python3.4 treebank_results.py treebank_d2v.bin "],shell=True)
