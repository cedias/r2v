#coding: utf-8

import subprocess

subprocess.call(["wget \"http://snap.stanford.edu/data/amazon/Shoes.txt.gz\""],shell=True)
subprocess.call(["python3.4 buildDatabase.py --gz Shoes.txt.gz amazonjson Shoes.db"],shell=True)
subprocess.call(["python3.4 db_to_R2V.py --min_count 100 Shoes.db Shoes_100 "],shell=True)
subprocess.call(["./d2v/d2v -train Shoes_100 -sentence-vectors 1 -size 200 -window 7 -cbow 0 -min-count 0 -sample 10e-4 -negative 5 -threads 5 -binary 1 -iter 25 -alpha 0.05 -output Shoes.d2v"],shell=True)
subprocess.call(["python3.4 predict_rating.py --k 5 --neigh item --mean_center Shoes.d2v Shoes.db"],shell=True)
