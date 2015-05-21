from gensim.models.doc2vec import Doc2Vec
import numpy as np

class R2VModel(object):

    def __init__(self, d2vModel):
        self.model = d2vModel
        self.model.init_sims()
        self._buildIndexs()

    @staticmethod
    def from_w2v_text(text,binary=True):
        d2v = Doc2Vec.load_word2vec_format(text,binary=binary)
        return R2VModel(d2v)

    def __getitem__(self, key):
        return self.model[key]

    def _buildIndexs(self):
        self.user_indexs = []
        self.item_indexs =  []
        self.review_indexs = []
        self.sent_indexs =  []
        self.word_indexs =  []
        self.rating_indexs =  []

        for i,word in enumerate(self.model.index2word): # O(n)
            if len(word) < 2 or word[1] != "_":
                self.word_indexs.append(i)
                continue
            elif word[0] == 'u':
                self.user_indexs.append(i)
                continue
            elif word[0] == 'i':
                if len(word.split("_")) == 2:
                    self.item_indexs.append(i)
                else:
                    self.review_indexs.append(i)
                continue
            elif word[0] == 'r':
                self.rating_indexs.append(i)
                continue
            elif word[0] == 's':
                self.sent_indexs.append(i)
                continue
            else:
                raise ValueError("Word {} not classified by indexer".format(word))


    def most_similar(self,vect, limit="all", topn=100):
        limits = {"all": True, "words": self.word_indexs, "sent": self.sent_indexs, "users": self.user_indexs,
                  "items": self.item_indexs, "rating": self.rating_indexs, "reviews": self.review_indexs}

        if limit not in limits.keys():
            print("limit not in {}".format(limits.keys()))
            return None

        if limit == "all":
            dist = np.dot(self.model.syn0norm, vect)
            best = np.argsort(dist)[::-1]

            if topn is not None:
                return [(self.model.index2word[sim], float(dist[sim])) for sim in best][1:topn + 1]
            else:
                return [(self.model.index2word[sim], float(dist[sim])) for sim in best]
        else:
            subset = self.model.syn0norm[limits[limit]]

            dist = np.dot(subset, vect)
            best = np.argsort(dist)[::-1]

            if topn is None:
                return [(self.model.index2word[limits[limit][sim]], float(dist[sim])) for sim in best]
            else:
                return [(self.model.index2word[limits[limit][sim]], float(dist[sim])) for sim in best][:topn]