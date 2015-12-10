#coding: utf-8

import subprocess

subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[10].pkl local/models/movies.d2v_[i10] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[9].pkl local/models/movies.d2v_[i9] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[8].pkl local/models/movies.d2v_[i8] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[7].pkl local/models/movies.d2v_[i7] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[6].pkl local/models/movies.d2v_[i6] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[5].pkl local/models/movies.d2v_[i5] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[4].pkl local/models/movies.d2v_[i4] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[3].pkl local/models/movies.d2v_[i3] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[2].pkl local/models/movies.d2v_[i2] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[1].pkl local/models/movies.d2v_[i1] /tmp/movies.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/movies_item_mc_UIR[0].pkl local/models/movies.d2v_[i0] /tmp/movies.db "],shell=True)
