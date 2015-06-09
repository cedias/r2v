#coding: utf-8

import argparse
from VectReco.R2VModel import R2VModel
from gensim.matutils import unitvec
import numpy as np

# BEST 40% ./d2v -train ../treebank.d2v -output ../treebank_d2v.bin -binary 1 -hs 0 -window 10 -sample 0 -min-count 0 -negative 15 -sentence-vectors 1 -cbow 0 -iter 10 -threads 6


def sump(list,k=1):
    sum_sim = 0
    sum_pond = 0
    for i,(n,sim) in enumerate(list):
        sim = (sim+1)/2
        sum_sim += sim
        sum_pond += n*sim
        if i == k-1:
            return round(sum_pond/sum_sim)

def ksim(uk_list,k):
    unknown_sump = [( sump(list_sim,k),real )for list_sim,real in uk_list]
    ok = sum([1 for pred,real in unknown_sump if pred == real])
    ko = sum([1 for pred,real in unknown_sump if pred != real])
    print("Sentiment Treebank {}-most-similar Fine-Grained Results: {} ok, {} ko, {} total ===>  {}% accuracy, {}% error".format(k,ok,ko,ok+ko,(ok/(ok+ko+0.0))*100,(ko/(ok+ko+0.0))*100))


def main(args):


    mod = R2VModel.from_w2v_text(args.model,binary=True)
    sentiments = np.array([unitvec(mod["sent_{}".format(i)]) for i in range(0,args.classes)])
    mod.set_cache(sentiments)

    unknown = [(mod.most_similar_cache_gen(mod[w]),int(w.split("_")[1])) for w in mod.model.index2word if w.split("_")[0] == "unk" ]

    for k in range(1,args.classes+1):
        ksim(unknown,k)


parser = argparse.ArgumentParser()
parser.add_argument("model",default="treebank_d2v.bin", type=str)
parser.add_argument("--classes",default=5,type=int)
args = parser.parse_args()

main(args)
