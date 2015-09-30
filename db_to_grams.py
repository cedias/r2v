#coding: utf8

import sqlite3
from random import shuffle
import argparse


def wordCount(db):
    words = {}
    for sent, labels in iterReviews(db):
        for word in sent:
            if word not in words:
                words[word] = 1
            else:
                words[word] += 1
    return words


def filterWords(words, min_count=0):
    return set(word for word in words if words[word] >= min_count)


def iterReviews(db):
    con = sqlite3.connect(db)
    c = con.cursor()
    i = 0

    for pid, uname, rev, rating in c.execute('SELECT item as pid, user as uname, review as rev, rating as rating FROM reviews WHERE NOT test'):
        if pid is None or uname is None or rating is None or rev is None:
            continue
        else:
            for sent in rev.split("."):
                if len(sent) < 2:
                    continue
                else:
                    yield (sent.split(" "), ['u_{}~i_{}'.format(uname,pid), 'r_{}'.format(rating)])
                    i += 1


def main(args):
    f = open(args.output, "w",encoding="utf-8")
    buff = []
    i = 0

    words = filterWords(wordCount(args.db), min_count=args.min_count)

    for sent, labels in iterReviews(args.db):

        sent = [word for word in sent if word in words]

        if len(sent) < args.min_sent_size:
            continue

        for label in labels:
            buff.append("{} {}\n".format(label, " ".join(sent)))
            i += 1

        if len(buff) >= args.buff_size:
            shuffle(buff)
            for se in buff:
                f.write(se)
            buff = []
            print("wrote {} sentences".format(i))

    shuffle(buff)
    for se in buff:
        f.write(se)
    f.close()

    print("wrote {} sentences".format(i))

parser = argparse.ArgumentParser()
parser.add_argument("db", type=str)
parser.add_argument("output", type=str, default="sentences.txt")
parser.add_argument("--min_count", type=int, default=100)
parser.add_argument("--min_sent_size", type=int, default=5)
parser.add_argument("--buff_size", type=int, default=1000000)
args = parser.parse_args()

main(args)
