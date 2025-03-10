import re
import random
import hashlib
import json
import gzip
import datetime
from random import shuffle

class DatasetIterator(object):

    def __init__(self,dataset,zipped=True,encoding="utf-8"):
        self.dataset = dataset
        self.itemPat = None
        self.userPat = None
        self.textPat = None
        self.ratingPat = None
        self.timePat = None
        self.reviewSep = None
        self.zipped = zipped
        self.encoding = encoding

    def split_getLast(self,text):
        split = text.split(" ", 1)
        if len(split) < 2:
            return None
        else:
            return split[1].strip()

    def preprocess(self,text,rating):

        regex_ponctuation = r"[\t,;#$^:*\[\]\+\|\{\}~%/\'=\"><\(\)_\-!&?]+"
        regex_double = r"([\.\.]+)"
        regex_br = r"<br />"
        regex_html = r"<.*?>"

        text = re.sub(regex_br, ".", text)
        text = re.sub(regex_html, " ", text)
        text = re.sub(regex_double, ".", text)
        text = re.sub(regex_ponctuation, " ", text)
        text = re.sub(r"(\s)+", " ", text)  # multi-spaces
        text = text.strip()

        return (text,float(rating))

    def iterate(self):
        user2id = {}
        item2id = {}
        dupeSet = set()
        dupeCount = 0
        item = None
        user = None
        text = None
        rating = None
        times = None
        i = 0

        if self.zipped:
            f = gzip.open(self.dataset, "r")
        else:
            f = open(self.dataset, "r",encoding=self.encoding)

        for line in f:
            line = line.decode(self.encoding,errors="ignore")
            if(self.itemPat.search(line)):
                val = self.split_getLast(line)

                if val is None:
                    continue
                else:
                    item = val
                    if item not in item2id:
                        iid = len(item2id) + 1
                        item2id[item] = iid
                    item = item2id[item]

            if(self.userPat.search(line)):
                val = self.split_getLast(line)

                if val is None:
                    continue
                else:
                    user = val
                    if user not in user2id:
                        uid = len(user2id) + 1
                        user2id[user] = uid
                    user = user2id[user]

            if(self.textPat.search(line)):
                val = self.split_getLast(line)
                if val is None:
                    continue
                else:
                    text = val

            if(self.ratingPat.search(line)):
                val = self.split_getLast(line)
                if val is None:
                    continue
                else:
                    rating = val

            if(self.timePat.search(line)):
                val = self.split_getLast(line)
                if val is None:
                    continue
                else:
                    times = val

            if(self.reviewSep.search(line)):
                if item is not None and user is not None and text is not None and rating is not None and times is not None:
                    hashedText = hashlib.md5(text.encode('utf-8')).hexdigest()

                    if hashedText in dupeSet:
                        dupeCount += 1
                    else:
                        dupeSet.add(hashedText)
                        if (random.random() > 0.80):
                            test = True
                        else:
                            test = False
                        text, rating = self.preprocess(text,rating)
                        yield((item, user, text, rating, times, test))

                    item = user = text = rating = times = None
                    i += 1
                    if i % 10000 == 0:
                        print("Imported {} reviews, found {} exact duplicates".format(i, dupeCount))
        print("Imported {} reviews, found {} exact duplicates".format(i, dupeCount))

class BeeradvocateIterator(DatasetIterator):
    def __init__(self,dataset,zipped=True,encoding="utf-8"):
        DatasetIterator.__init__(self,dataset,zipped=True,encoding="utf-8")
        self.itemPat = re.compile('^beer/beerId:')
        self.userPat = re.compile('^review/profileName:')
        self.textPat = re.compile('^review/text:')
        self.ratingPat = re.compile('^review/overall:')
        self.timePat = re.compile('^review/time:')
        self.reviewSep = re.compile('^$')

class AmazonIterator(DatasetIterator):
    def __init__(self,dataset,zipped=True,encoding="utf-8"):
        DatasetIterator.__init__(self,dataset,zipped=True,encoding="utf-8")
        self.itemPat = re.compile('^product/productId:')
        self.userPat = re.compile('^review/userId:')
        self.textPat = re.compile('^review/text:')
        self.ratingPat = re.compile('^review/score:')
        self.timePat = re.compile('^review/time:')
        self.reviewSep = re.compile('^$')

class RatebeerIterator(BeeradvocateIterator):
    def __init__(self,dataset,zipped=True,encoding="utf-8"):
        BeeradvocateIterator.__init__(self,dataset,zipped=True,encoding="utf-8")

    def preprocess(self,text,rating):
        regex_ponctuation = r"[\t,;#$^:*\[\]\+\|\{\}~%/\'=\"><\(\)_\-!&?]+"
        regex_double = r"([\.\.]+)"
        regex_br = r"<br />"
        regex_html = r"<.*?>"

        text = re.sub(regex_br, ".", text)
        text = re.sub(regex_html, " ", text)
        text = re.sub(regex_double, ".", text)
        text = re.sub(regex_ponctuation, " ", text)
        text = re.sub(r"(\s)+", " ", text)  # multi-spaces
        text = text.strip()

        rating = rating.split("/")
        rating = float(rating[0])/float(rating[1])*5

        return (text,float(rating))

class YelpIterator(DatasetIterator):
    def __init__(self,dataset,zipped=True,encoding="utf-8"):
        if zipped:
            print("PLEASE UNZIP")

        DatasetIterator.__init__(self,dataset,zipped=True,encoding="utf-8")

    def iterate(self):
        f = open(self.dataset)
        user2id = {}
        item2id = {}
        i=0
        dupeCount = 0
        dupeSet = set()

        for line in f:
            dec = json.loads(line)

            text = dec["text"]
            rating = dec["stars"]
            times = "15545423445"
            if dec["business_id"] not in item2id:
                item2id[dec["business_id"]] = len(item2id) + 1

            if dec["user_id"] not in user2id:
                user2id[dec["user_id"]] = len(user2id) + 1

            item = item2id[dec["business_id"]]
            user = user2id[dec["user_id"]]

            hashedText = hashlib.md5(text.encode('utf-8')).hexdigest()

            if hashedText in dupeSet:
                dupeCount += 1
            else:
                dupeSet.add(hashedText)
                if (random.random() > 0.80):
                    test = True
                else:
                    test = False

                text, rating = self.preprocess(text,rating)

                yield((item, user, text, rating, times, test))

            item = user = text = rating = times = None
            i += 1
            if i % 10000 == 0:
                print("Imported {} reviews, found {} exact duplicates".format(i, dupeCount))
            #item != user != text != rating != times


class AmazonJsonIterator(DatasetIterator):
    """
    {
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}

    """
    def __init__(self,dataset,zipped=True,encoding="utf-8"):
        DatasetIterator.__init__(self,dataset,zipped=True,encoding="utf-8")

    def iterate(self):
        f = gzip.open(self.dataset)
        user2id = {}
        item2id = {}
        i=0
        dupeCount = 0
        dupeSet = set()

        for line in f:
            dec = eval(line)

            text = dec["reviewText"]
            rating = dec["overall"]
            times = "15545423445"
            if dec["asin"] not in item2id:
                item2id[dec["asin"]] = len(item2id) + 1

            if dec["reviewerID"] not in user2id:
                user2id[dec["reviewerID"]] = len(user2id) + 1

            item = item2id[dec["asin"]]
            user = user2id[dec["reviewerID"]]

            hashedText = hashlib.md5(text.encode('utf-8')).hexdigest()

            if hashedText in dupeSet:
                dupeCount += 1
            else:
                dupeSet.add(hashedText)
                if (random.random() > 0.80):
                    test = True
                else:
                    test = False

                text, rating = self.preprocess(text,rating)

                yield((item, user, text, rating, times, test))


            i += 1
            if i % 10000 == 0:
                print("Imported {} reviews, found {} exact duplicates".format(i, dupeCount))



class BufferedShuffledIterator(object):

    def __init__(self,size,iterator):
        self.len = size
        self.iterator = iterator

    def iterate(self):
        buff = []
        for a in self.iterator.iterate():
            buff.append(list(a))

            if len(buff) >= self.len:
                shuffle(buff)
                for b in buff:
                    if random.random()>0.8:
                        b[5] = 1
                    else:
                        b[5] = 0
                    yield(tuple(b))
                buff = []

        shuffle(buff)
        for b in buff:
            if random.random()>0.8:
                b[5] = 1
            else:
                b[5] = 0
            yield(tuple(b))
        buff = []

