#coding: utf-8

import subprocess

subprocess.call(["wget \"http://95.85.49.48/ratebeer.tar.gz\""],shell=True)
subprocess.call(["python3.4 buildDatabase.py --gz ratebeer.tar.gz ratebeer ratebeer.db"],shell=True)
subprocess.call(["python3.4 db_to_R2V.py --min_count 10000 ratebeer.db ratebeer-10k.txt "],shell=True)
subprocess.call(["./d2v/d2v -train ratebeer-10k.txt -sentence-vectors 1 -size 200 -window 10 -cbow 0 -min-count 0 -sample 10e-4 -negative 5 -threads 5 -binary 1 -iter 1 -alpha 0.08 -output rb.d2v"],shell=True)
subprocess.call(["python3.4 predict_rating.py --k 25 --neigh user --mean_center rb.d2v ratebeer.db"],shell=True)
