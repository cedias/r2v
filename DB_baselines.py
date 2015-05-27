# coding: utf-8
import gensim.corpora
import PoussCB.data
import PoussCB.biases
import PoussCB.evaluations
import PoussCB.collaborative_filtering
import PoussCB.texts
import PoussCB.unified_recsys
from VectReco.Database import Database
from random import shuffle
import argparse

def load_data(filename):
    print("Loading data")
    db = Database(filename)
    training_reviews, test_reviews = PoussCB.data.load_from_database(db, load_texts=True)
    shuffle(training_reviews)
    training_reviews, validation_reviews = PoussCB.data.split_sets(training_reviews)
    evaluation = PoussCB.evaluations.RmseEvaluation(training_reviews, validation_reviews, test_reviews)
    return evaluation, training_reviews, validation_reviews


def run_overall_bias(training_reviews):
    print("\nOverall bias")
    overall_bias = PoussCB.biases.OverallBias()
    overall_bias.fit(training_reviews)
    return overall_bias


def run_user_bias(training_reviews):
    print("\nUser bias")
    user_bias = PoussCB.biases.UserBias()
    user_bias.fit(training_reviews)
    return user_bias


def run_item_bias(training_reviews):
    print("\nItem bias")
    item_bias = PoussCB.biases.ItemBias()
    item_bias.fit(training_reviews)
    return item_bias


def run_collaborative_filtering(training_reviews, validation_reviews, k, epochs, eta_0, l2_weight):
    print("\nCollaborative filtering")
    colfil = PoussCB.collaborative_filtering.StochasticGradientMatrixFactorization(k, epochs, eta_0, l2_weight)
    colfil.fit_with_validation(training_reviews, validation_reviews)
    return colfil


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



if __name__ == u"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str)
    parser.add_argument("--latent",default=5, type=float)
    parser.add_argument("--epochs",default=100, type=float)
    parser.add_argument("--alpha",default=0.01, type=float)
    parser.add_argument("--reg", default=0.01, type=float)
    args = parser.parse_args()
    k = args.latent
    # Collaborative filtering
    cf_epochs = args.epoch
    cf_eta_0 = args.alpha
    cf_l2_weight = args.reg
    db = args.db
    run_all(db, k, cf_epochs, cf_eta_0, cf_l2_weight)