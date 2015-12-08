#coding: utf-8

import subprocess

subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[0].pkl /local/dias/models/ratebeer_classic.d2v_[i0] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[1].pkl /local/dias/models/ratebeer_classic.d2v_[i1] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[2].pkl /local/dias/models/ratebeer_classic.d2v_[i2] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[3].pkl /local/dias/models/ratebeer_classic.d2v_[i3] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[4].pkl /local/dias/models/ratebeer_classic.d2v_[i4] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[5].pkl /local/dias/models/ratebeer_classic.d2v_[i5] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[6].pkl /local/dias/models/ratebeer_classic.d2v_[i6] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[7].pkl /local/dias/models/ratebeer_classic.d2v_[i7] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[8].pkl /local/dias/models/ratebeer_classic.d2v_[i8] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[9].pkl /local/dias/models/ratebeer_classic.d2v_[i9] /tmp/ratebeer.db "],shell=True)
subprocess.call(["OMP_NUM_THREADS=4 python3.4 predict_rating.py --k 100 --neigh item --mean_center --output /local/dias/ratebeer_item_mc_UIR[10].pkl /local/dias/models/ratebeer_classic.d2v_[i10] /tmp/ratebeer.db "],shell=True)
