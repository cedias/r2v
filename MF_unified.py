# coding: utf-8
import gensim.corpora
import ClassicCB.data
import ClassicCB.biases
import ClassicCB.evaluations
import ClassicCB.collaborative_filtering
import ClassicCB.texts
import ClassicCB.unified_recsys
from VectReco.Database import Database
from random import shuffle
import argparse

def load_data(filename):
    print("Loading data")
    db = Database(filename)
    training_reviews, test_reviews = ClassicCB.data.load_from_database(db, load_texts=True)
    shuffle(training_reviews)
    training_reviews, validation_reviews = ClassicCB.data.split_sets(training_reviews)
    evaluation = ClassicCB.evaluations.RmseEvaluation(training_reviews, validation_reviews, test_reviews)
    return evaluation, training_reviews, validation_reviews


def run_overall_bias(training_reviews):
    print("\nOverall bias")
    overall_bias = ClassicCB.biases.OverallBias()
    overall_bias.fit(training_reviews)
    return overall_bias


def run_user_bias(training_reviews):
    print("\nUser bias")
    user_bias = ClassicCB.biases.UserBias()
    user_bias.fit(training_reviews)
    return user_bias


def run_item_bias(training_reviews):
    print("\nItem bias")
    item_bias = ClassicCB.biases.ItemBias()
    item_bias.fit(training_reviews)
    return item_bias


def run_collaborative_filtering(training_reviews, validation_reviews, k, epochs, eta_0, l2_weight):
    print("\nCollaborative filtering")
    colfil = ClassicCB.collaborative_filtering.StochasticGradientMatrixFactorization(k, epochs, eta_0, l2_weight)
    colfil.fit_with_validation(training_reviews, validation_reviews)
    return colfil


def extract_dictionary(training_reviews, no_below, no_above, keep_n):
    print("\nDictionary")
    dictionary = gensim.corpora.Dictionary()
    dictionary.add_documents([review.get_text() for review in training_reviews])
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    return dictionary


def run_topic_extraction(training_reviews, dictionary, k, epochs):
    print("\nTopic extraction")
    topic_extractor = ClassicCB.texts.TopicExtractor(dictionary, k, epochs)
    topic_extractor.fit_transform(training_reviews)
    return topic_extractor


def run_unified(training_reviews, validation_reviews, k, colfil, topic_extractor, epochs, eta_0, l2_weight, balance):
    print("\nUnified model")
    unified = ClassicCB.unified_recsys.UnifiedRecsys(k, epochs, eta_0, l2_weight, balance)
    unified.warm_restart(colfil, topic_extractor)
    unified.fit_with_validation(training_reviews, validation_reviews)
    return unified


def run_all(filename, k, cf_epochs, cf_eta_0, cf_l2_weight):
    evaluation, training_reviews, validation_reviews = load_data(filename)
    overall_bias = run_overall_bias(training_reviews)
    print("RMSE overall bias: {} ".format(evaluation.evaluate(overall_bias)))
    user_bias = run_user_bias(training_reviews)
    print("RMSE user bias: {} ".format(evaluation.evaluate(user_bias)))
    item_bias = run_item_bias(training_reviews)
    print("RMSE item bias: {} ".format(evaluation.evaluate(item_bias)))
    colfil = run_collaborative_filtering(training_reviews, validation_reviews, k, cf_epochs, cf_eta_0, cf_l2_weight)
    print("RMSE item bias: {}".format(evaluation.evaluate(colfil)))
    # Dictionary
    dictionary = extract_dictionary(training_reviews, no_below, no_above, keep_n)
    # Topic extraction
    topic_extractor = run_topic_extraction(training_reviews, dictionary, k, te_epochs)
    # Unified
    unified = run_unified(training_reviews, validation_reviews, k, colfil, topic_extractor, uni_epochs, uni_eta_0,
                          uni_l2_weight, uni_balance)
    print("RMSE item bias: %.5f (training) %.5f (validation) %.5f (test)" % (evaluation.evaluate(unified)))


if __name__ == u"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str)
    parser.add_argument("--latent",default=5, type=float)
    parser.add_argument("--epochs",default=100, type=float)
    parser.add_argument("--alpha",default=0.01, type=float)
    parser.add_argument("--reg", default=0.01, type=float)
    parser.add_argument("--keep_n", default=0.01, type=float)
    parser.add_argument("--no_below", default=0.01, type=float)
    parser.add_argument("--no_above", default=0.01, type=float)

    args = parser.parse_args()
    k = args.latent
    # Collaborative filtering
    cf_epochs = args.epochs
    cf_eta_0 = args.alpha
    cf_l2_weight = args.reg
    db = args.db
    run_all(db, k, cf_epochs, cf_eta_0, cf_l2_weight)