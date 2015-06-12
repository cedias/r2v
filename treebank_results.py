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


def ksim_sents(kn_list):
    ok = sum([1 for pred,real in kn_list if pred == real])
    ko = sum([1 for pred,real in kn_list if pred != real])
    print("Sentiment Treebank 1-most-similar sent Fine-Grained Results: {} ok, {} ko, {} total ===>  {}% accuracy, {}% error".format(ok,ko,ok+ko,(ok/(ok+ko+0.0))*100,(ko/(ok+ko+0.0))*100))

def main(args):
    s_sim = args.near_sent

    mod = R2VModel.from_w2v_text(args.model,binary=True)
    if s_sim:
        sents = np.array([unitvec(mod[x]) for x in mod.model.vocab if x.split("_")[0]=="kn"])
        sents_word = [int(x.split("_")[1]) for x in mod.model.vocab if x.split("_")[0]=="kn"]
        mod.set_cache(sents)
        unknown = [(mod.most_similar_cache_gen(mod[w],1),int(w.split("_")[1])) for w in mod.model.index2word if w.split("_")[0] == "unk" ]
        unknown = [(sents_word[ls[0][0]],pred) for ls ,pred in unknown]
        ksim_sents(unknown)


    else:
        sentiments = np.array([unitvec(mod["sent_{}".format(i)]) for i in range(0,args.classes)])
        mod.set_cache(sentiments)
        unknown = [(mod.most_similar_cache_gen(mod[w]),int(w.split("_")[1])) for w in mod.model.index2word if w.split("_")[0] == "unk" ]
        for k in range(1,args.classes+1):
            ksim(unknown,k)


parser = argparse.ArgumentParser()
parser.add_argument("model",default="treebank_d2v.bin", type=str)
parser.add_argument("--classes",default=5,type=int)
parser.add_argument("--near_sent",default=False,type=bool)
args = parser.parse_args()

main(args)
