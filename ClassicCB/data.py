# coding=utf-8
import gzip


class Review:
    u"""
    Defines a review as a tuple user, item, rating, timestamp.
    Each parameter can be accessed with a named getter function.
    """
    def __init__(self, user, item, rating, timestamp):
        self.user = user
        self.item = item
        self.rating = rating
        self.timestamp = timestamp

    def get_user(self):
        return self.user

    def get_item(self):
        return self.item

    def get_rating(self):
        return self.rating

    def get_timestamp(self):
        return self.timestamp


class TextReview(Review):
    u"""
    Defines text reviews: they contain, in addition to the other fields of a review, a text that explains the rating.
    """
    def __init__(self, user, item, rating, timestamp, text):
        Review.__init__(self, user, item, rating, timestamp)
        self.text = text

    def get_text(self):
        return self.text


def load_simple_gz(filename, load_texts=False):
    u"""
    Utility function that loads reviews from the given gzip file with one review/line: user item rating timestamp nb_words review_text.
    :param filename: the path to the file to load (must be a gzip file).
    :param load_texts: if True, then texts are loaded along with other parameters.
    :return: the list of reviews stored in the file.
    """
    reviews = []
    with gzip.open(filename, u"rb") as data_file:
        for line in data_file:
            tokens = line.decode(u"UTF-8", u"ignore").strip().lower().split()
            user, item, rating = tokens[:3]
            timestamp = tokens[3]
            if load_texts:
                nb_words = int(tokens[4])
                words = tokens[5:]
                assert nb_words == len(words), u"<%d> words when <%d> were expected" % (len(words), nb_words)
                reviews.append(TextReview(user, item, float(rating), int(timestamp), words))
            else:
                reviews.append(Review(user, item, float(rating), int(timestamp)))
    return reviews


def load_from_database(db, load_texts=False):
    u"""
    Utility function that loads train and test reviews from a given database
    :param filename: the path to the file to load (must be a gzip file).
    :param load_texts: if True, then texts are loaded along with other parameters.
    :return: the list of reviews stored in the file.
    """
    train_reviews = []
    test_reviews = []
    for item,user,review,rating,timestamp,test  in db.getFullReviews():

        if test == 0:
            if load_texts:
                train_reviews.append(TextReview(user, item, float(rating), int(timestamp), review))
            else:
                train_reviews.append(Review(user, item, float(rating), int(timestamp)))
        else:
            if load_texts:
                test_reviews.append(TextReview(user, item, float(rating), int(timestamp), review))
            else:
                test_reviews.append(Review(user, item, float(rating), int(timestamp)))

    return train_reviews,test_reviews


def rescale_ratings(reviews):
    u"""
    Utility function that rescales in place the ratings from [1, 5] to [-1, 1].
    :param reviews: list of reviews.
    :return: nothing, modification is done in place.
    """
    # 1 - 5 => -1 - 1
    for review in reviews:
        review.rating = review.rating / 2.0 - 1.5


def split_sets(reviews, training=0.85):
    """
    Utility function to split reviews into two sets using the given proportion.
    :param reviews: a list a reviews
    :param training: proportion (percentage) of reviews to put in the training set.
    :return: two lists of reviews: the training and validation sets.
    """
    m_training = int(training * len(reviews))
    return reviews[:m_training], reviews[m_training:]