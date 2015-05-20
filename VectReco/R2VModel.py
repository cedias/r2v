from gensim.models.doc2vec import Doc2Vec
import numpy as np

class R2VModel(object):

    def __init__(self, d2vModel):
        self.model = d2vModel
        self.model.init_sims()
        self._buildIndexs()

    def _buildIndex(self):
        self.user_indexs = [i for i, word in enumerate(self.model.index2word) if len(word) > 2 and word[0] == "u" and word[1] == "_"]
        self.item_indexs = [i for i, word in enumerate(self.model.index2word) if len(word) > 2 and word[0] == "i" and word[1] == "_" and len(word.split("_")) == 2 ]
        self.review_indexs = [i for i, word in enumerate(self.model.index2word) if len(word) > 2 and word[0] == "i" and word[1] == "_" and len(word.split("_")) > 2 ]
        self.sent_indexs = [i for i, word in enumerate(self.model.index2word) if len(word) > 2 and word[0] == "s" and word[1] == "_"]
        self.word_indexs = [i for i, word in enumerate(self.model.index2word) if len(word) < 2 or word[1] != "_"]
        self.rating_indexs = [i for i, word in enumerate(self.model.index2word) if len(word) > 2 and word[0] == "r" and word[1] == "_"]

        for i,word in enumerate(self.model.index2word):
            if len(word) < 2 or word[1] != "_":
                word.indexs.append()

        pass

    def most_similar(model,vect, limit="all", topn=100):
        pass
        # limits = {"all": True, "words": True, "sent": "s", "users": "u", "items": "i", "rating": "r", "reviews": True}
        # if limit not in limits.keys():
        #     print("limit not in {}".format(limits.keys()))
        #     return None
        #
        # if limit == "all":
        #     dist = np.dot(model.syn0norm, vect)
        #     best = np.argsort(dist)[::-1]
        #
        #     if topn is not None:
        #         return [(model.index2word[sim], float(dist[sim])) for sim in best][1:topn + 1]
        #     else:
        #         return [(model.index2word[sim], float(dist[sim])) for sim in best]
        #
        # if limit == "rating":
        #     if topn is not None:
        #         return [(model.index2word[sim], float(dist[sim])) for sim in best if len(model.index2word[sim]) > 2 and model.index2word[sim][1] == '_' and model.index2word[sim][0] == limits[limit]][0:topn]
        #     else:
        #         return [(model.index2word[sim], float(dist[sim])) for sim in best if len(model.index2word[sim]) > 2 and model.index2word[sim][1] == '_'and model.index2word[sim][0] == limits[limit]]
        #
        #
        # if limit == "words":
        #     if topn is not None:
        #         return [(model.index2word[sim], float(dist[sim])) for sim in best if len(model.index2word[sim]) < 2 or model.index2word[sim][1] != '_'][1:topn + 1]
        #     else:
        #         return [(model.index2word[sim], float(dist[sim])) for sim in best if len(model.index2word[sim]) < 2 or model.index2word[sim][1] != '_']
        #
        # else:
        #     if topn is not None:
        #         return [(model.index2word[sim], float(dist[sim])) for sim in best if len(model.index2word[sim]) > 2 and model.index2word[sim][1] == '_' and model.index2word[sim][0] == limits[limit]][1:topn + 1]
        #     else:
        #         return [(model.index2word[sim], float(dist[sim])) for sim in best if len(model.index2word[sim]) > 2 and model.index2word[sim][1] == '_'and model.index2word[sim][0] == limits[limit]]
        #
        #
        #
