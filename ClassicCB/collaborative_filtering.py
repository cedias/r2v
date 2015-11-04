# coding=utf-8
import numpy as np

from ClassicCB.biases import OverallBias
from ClassicCB.evaluations import RmseEvaluation
import pyximport; pyximport.install()
import ClassicCB.update_profiles as up


class StochasticGradientMatrixFactorization(OverallBias):
    u"""
    This is an implementation of the collaborative filtering model defined in Koren's publication:

    Matrix Factorization Techniques for Recommender Systems

    The model is a factorisation of the partially observed rating matrix: R = (R_{ui})_{ui} as :
    min sum_{ui} (R_{ui} - f(u, i))^2 + \lambda (|b_u|_2^2 + |b_i|_2^2 + |p_u|_2^2 + |q_i|_2^2)
    with f(u, i) = \mu + b_u + b_i + < p_u, q_i >

    The implementation uses a stochastic gradient descent on observed ratings with a decreasing learning rate and
    a L2 regularization.
    """
    def __init__(self, k=5, epochs=100, eta_0=0.01, l2_weight=0.01, validation=0.2):
        """
        Builds a new model with given training parameters.
        :param k: the number of hidden dimension to learn the profiles. This is the inner dimension of the matrix
        factorization.
        :param epochs: the number of training epochs.
        :param eta_0: the initial learning rate.
        :param l2_weight: the weight of the L2 regularization on the factors (biases and latent factors).
        :param validation: is given and in [0, 1[, a validation set is extracted from training reviews to estimate
        the generalisation of the model. This parameter controls the size, in percent, of the validation set w.r.t.
        the given set of training reviews.
        :return: nothing.
        """
        OverallBias.__init__(self)
        self.users = {}
        self.items = {}
        self.k = k
        self.epochs = epochs
        self.eta_0 = eta_0
        self.l2_weight = l2_weight
        self.validation = validation
        self.learning_rate = self.eta_0

    def initialize(self, training_reviews):
        u"""
        Initializes the profiles of users and items to random initial values.
        No fanning out nor scaling is used for now.
        :param training_reviews: the training reviews.
        :return: nothing, the profiles are created in place.
        """
        for review in training_reviews:
            user, item = review.get_user(), review.get_item()
            if user not in self.users:
                self.users[user] = (np.random.randn(), np.random.randn(self.k))
            if item not in self.items:
                self.items[item] = (np.random.randn(), np.random.randn(self.k))

    def update_profiles(self, user, item, delta, b_u, b_i, gamma_u, gamma_i):
        u"""
        Helper function that implements the update rules.
        :param user: user index.
        :param item: item index.
        :param delta: delta between the actual and predicted ratings.
        :param b_u: user bias.
        :param b_i: item bias.
        :param gamma_u: user latent profile.
        :param gamma_i: item latent profile.
        :return: nothing, modification are made in place.
        """
        self.users[user] = (b_u - self.learning_rate * (delta + self.l2_weight * b_u),
                            gamma_u - self.learning_rate * (delta * gamma_i + self.l2_weight * gamma_u))
        self.items[item] = (b_i - self.learning_rate * (delta + self.l2_weight * b_i),
                            gamma_i - self.learning_rate * (delta * gamma_u + self.l2_weight * gamma_i))


    def one_epoch(self, training_reviews):
        u"""
        Helper function that implements one epoch: a pass on the training reviews.
        Reviews are shuffled into random order and each is seen once.
        :param training_reviews: the training reviews.
        :return: nothing, parameters are updated in place.
        """

        arr = np.arange(len(training_reviews))
        np.random.shuffle(arr)
        for index in arr:
            review = training_reviews[index]
            user, item, rating = review.get_user(), review.get_item(), review.get_rating()
            b_u, gamma_u = self.users[user]
            b_i, gamma_i = self.items[item]
            delta = self.overall_bias + b_u + b_i + np.dot(gamma_u, gamma_i) - rating
            up.update_profiles_cy(self,user,item, delta, b_u, b_i, gamma_u, gamma_i)
            #self.update_profiles(user, item, delta, b_u, b_i, gamma_u, gamma_i)



    def fit(self, training_reviews):
        u"""
        Trains the model using set parameters and given training reviews.
        If validation is used, then the training reviews are split in two sets: one training and one validation set.
        :param training_reviews: training reviews to use.
        :return: nothing.
        """
        if self.validation is not None and 0 < self.validation < 1:
            m = len(training_reviews)
            indexes = np.arange(m)
            np.random.shuffle(indexes)
            split_index = int(self.validation * m)
            training_indexes, validation_indexes = indexes[:split_index], indexes[split_index:]
            new_training_reviews = [training_reviews[i] for i in training_indexes]
            new_validation_reviews = [training_reviews[i] for i in validation_indexes]
            self.fit_with_validation(new_training_reviews, new_validation_reviews)
        else:
            self.fit_with_validation(training_reviews, None)

    def fit_with_validation(self, training_reviews, validation_reviews):
        u"""
        Fits the model using the given training reviews and validation reviews. Here, even if validation is set from
        the constructor, it is ignored are the sets are explicitly given.
        :param training_reviews: the training reviews.
        :param validation_reviews: the validation reviews.
        :return: nothing.
        """
        OverallBias.fit(self, training_reviews)
        self.initialize(training_reviews)
        evaluation = RmseEvaluation(training_reviews, validation_reviews, None)

        training_rmse, validation_rmse, test_rmse = evaluation.evaluate(self)
        print("RMSE @ epoch 0: {} (training), {} (validation), {} (test)".format( training_rmse, validation_rmse,test_rmse))

        for epoch in range(1, self.epochs + 1):
            self.learning_rate = self.eta_0 / float(1 + epoch)
            self.one_epoch(training_reviews)
            training_rmse, validation_rmse, test_rmse = evaluation.evaluate(self)
            print("RMSE @ epoch {}: {} (training), {} (validation)".format(epoch, training_rmse, validation_rmse))

    def predict(self, user, item):
        u"""
        Computes the prediction for the couple (user, item): f(user=u, item=i) = \mu + b_u + b_i + < p_u, q_i >
        :param user: user index.
        :param item: item index.
        :return:
        """
        if user in self.users and item in self.items:
            b_u, gamma_u = self.users[user]
            b_i, gamma_i = self.items[item]
            return self.overall_bias + b_u + b_i + np.dot(gamma_u, gamma_i)
        else:
            return OverallBias.predict(self, user, item)