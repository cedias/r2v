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

        for mc in [10,100,1000,2500,5000,7500]:
            classic_path += "[{}]".format(mc)
            subprocess.call(["python3.4 db_to_R2V.py --buff_size 1000000000 --min_count {} {} {}".format(mc,db_path,classic_path)],shell=True)


    print("TRAINING")

    for db in dbfiles:
        print("Training {}".format(db))

        db_path = join(mypath+"/db",db)
        classic_path = join(mypath+"/train_files",db.split('.')[0]+"_train_classic.txt")
        classic_model = join(mypath+"/models",db.split('.')[0]+"_classic.d2v")


        for mc in [10,100,1000,2500,5000,7500]:
            classic_path += "[{}]".format(mc)
            classic_model += "[{}]".format(mc)
            subprocess.call(["./d2v/d2v -train {} -sentence-vectors 1 -size 200 -window 5 -multi 1 -min-count 0 -sample 1e-5 -negative 5 -threads 6 -binary 1 -iter 10 -alpha 0.05 -output {}".format(classic_path,classic_model)],shell=True)

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
args = parser.parse_args()
main(args)
