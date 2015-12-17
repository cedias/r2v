from os import listdir
from os.path import isfile, join
import argparse
import subprocess

def main(args):
    mypath = args.path

    dbfiles = [ f for f in listdir(mypath+"/db") if isfile(join(mypath+"/db",f))]

    print(dbfiles)

    print("BUILDING TRAIN_SET")

    for db in dbfiles:
        print("Building {}".format(db))

        db_path = join(mypath+"/db",db)
        classic_path = join(mypath+"/train_files",db.split('.')[0]+"_train_classic.txt")
        sum_path = join(mypath+"/train_files",db.split('.')[0]+"_train_sum.txt")
        sum_rating_path = join(mypath+"/train_files",db.split('.')[0]+"_train_sum_rating.txt")

        subprocess.call(["python3.4 db_to_R2V.py --buff_size 1000000000 --min_count 10{} {}".format(db_path,classic_path)],shell=True)
        #subprocess.call(["python3.4 db_to_grams.py --buff_size 1000000000 --rating --min_count 10000 {} {}".format(db_path,sum_rating_path)],shell=True)
        #subprocess.call(["python3.4 db_to_grams.py --buff_size 1000000000 --min_count 100000 {} {}".format(db_path,sum_path)],shell=True)

    print("TRAINING")

    for db in dbfiles:
        print("Training {}".format(db))

        db_path = join(mypath+"/db",db)
        classic_path = join(mypath+"/train_files",db.split('.')[0]+"_train_classic.txt")
        sum_path = join(mypath+"/train_files",db.split('.')[0]+"_train_sum.txt")
        sum_rating_path = join(mypath+"/train_files",db.split('.')[0]+"_train_sum_rating.txt")
        classic_model = join(mypath+"/models",db.split('.')[0]+"_classic.d2v")
        sum_model = join(mypath+"/models",db.split('.')[0]+"sum.d2v")
        sum_rating_model = join(mypath+"/models",db.split('.')[0]+"_sum_rating.d2v")

        #print(db_path);print(classic_path);print(sum_path);print(sum_rating_path);print(classic_model);print(sum_model);print(sum_rating_model);

        subprocess.call(["./d2v/d2v -train {} -sentence-vectors 1 -size 200 -window 5 -multi 1 -min-count 0 -sample 1e-5 -negative 5 -threads 6 -binary 1 -iter 10 -alpha 0.05 -output {}".format(classic_path,classic_model)],shell=True)
        #subprocess.call(["./d2v/d2v -train {} -sentence-vectors 1 -size 200 -window 5 -sample 1e-5 -negative 5 -threads 6 -binary 1 -iter 10 -alpha 0.05 -output {}".format(sum_path,sum_model)],shell=True)
        #subprocess.call(["./d2v/d2v -train {} -sentence-vectors 1 -size 200 -window 5 -sample 1e-5 -negative 5 -threads 6 -binary 1 -iter 10 -alpha 0.05 -output {}".format(sum_rating_path,sum_rating_model)],shell=True)

        #./d2v/d2v -train /local/dias/ratebeer_10.txt -sentence-vectors 1 -size 600 -window 10  -sample 1e-4 -negative 5 -threads 4 -binary 1 -iter 20 -alpha 0.05 -output /local/dias/rb_600_20.d2v

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
args = parser.parse_args()
main(args)
