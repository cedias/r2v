# coding=utf-8
import numpy as np

from PoussCB.biases import OverallBias
from PoussCB.collaborative_filtering import StochasticGradientMatrixFactorization
from PoussCB.evaluations import RmseEvaluation
from PoussCB.texts import TopicExtractor


class UnifiedRecsys(StochasticGradientMatrixFactorization, TopicExtractor):
    u"""
    Unified recommender systems that uses a collaborative filtering part and a topic extraction part.
    """

    def __init__(self, k=5, epochs=100, eta_0=0.01, l2_weight=0.01, balance=0.1, validation=0.2, dictionary=None):
        u"""
        Sets the parameters of the unified recommender system model.
        :param k: latent dimension both for the collaborative filtering part and the topic extraction part.
        :param epochs: training epochs for both models.
        :param eta_0: initial learning rate for the collaborative filtering.
        :param l2_weight: weight of the L2 regularization for the collaborative filtering engine.
        :param balance: balance coefficient between both parts (collaborative filtering and topic extraction).
        :param validation: proportion of training data (percentage) to use as a validation set
        :param dictionary: a gensim dictionary to use for text to bag of word tranformations.
        :return:
        """
        StochasticGradientMatrixFactorization.__init__(self, k, epochs, eta_0, l2_weight, validation)
        self.balance = balance
        self.dictionary = dictionary
        if dictionary is None:
            self.topics = None
        else:
            self.topics = np.zeros((len(dictionary), k))
        self.using_warm_restart = False

    def warm_restart(self, recommender_system, topic_extractor):
        u"""
        Utility function that copies the state of given models to this model. Training parameters (learning rate, ...)
        are not modified.
        :param recommender_system: a collaborative filtering model.
        :param topic_extractor: a topic extraction model.
        :return: nothing, parameters are changed in place.
        """
        k, epochs, eta_0, l2_weight, balance = self.k, self.epochs, self.eta_0, self.l2_weight, self.balance
        assert isinstance(recommender_system, StochasticGradientMatrixFactorization)
        assert isinstance(topic_extractor, TopicExtractor)
        assert recommender_system.k == topic_extractor.k
        for key, value in recommender_system.__dict__.items():
            self.__dict__[key] = value
        for key, value in topic_extractor.__dict__.items():
            self.__dict__[key] = value
        self.using_warm_restart = True
        self.k, self.epochs, self.eta_0, self.l2_weight, self.balance = k, epochs, eta_0, l2_weight, balance

    def update_unified_profiles(self, user, item, delta_rating, b_u, b_i, gamma_u, gamma_i, delta_unified, theta):
        u"""
        Utility function that updates parameters of the collaborative filtering part.
        :param user: user index.
        :param item: item index.
        :param delta_rating: delta between predicted and actual ratings.
        :param b_u: user bias.
        :param b_i: item bias.
        :param gamma_u: latent profile of the user.
        :param gamma_i: latent profile of the item.
        :param delta_unified: loss of the unified model.
        :param theta: topic distribution of current text.
        :return: nothing, modifications are done in place.
        """
        self.users[user] = (b_u - self.learning_rate * (delta_rating + self.l2_weight * b_u),
                            gamma_u - self.learning_rate * (
                                delta_rating * gamma_i + self.balance * delta_unified * theta + self.l2_weight * gamma_u))
        self.items[item] = (b_i - self.learning_rate * (delta_rating + self.l2_weight * b_i),
                            gamma_i - self.learning_rate * (delta_rating * gamma_u + self.l2_weight * gamma_i))

    def update_topics_stochastic(self, delta_topics, theta):
        u"""
        Stochastic update of the topic matrix and topic distribution for this text.
        :param delta_topics: loss of the topic extraction model.
        :param theta: topic distribution.
        :return: nothing, modification is done in place.
        """
        self.topics -= self.learning_rate * (np.outer(delta_topics, theta) + self.l2_weight * self.topics)
        self.topics[self.topics < 0] = 0

    def one_unified_epoch(self, training_reviews, training_texts, training_thetas):
        u"""
        Utility function that trains the model for one epoch: one pass on all training examples.
        :param training_reviews: training reviews.
        :param training_texts: text matrix of the training reviews.
        :param training_thetas: topic distribution matrix for the training reviews.
        :return: nothing, modifications are done in place.
        """
        arr = np.arange(len(training_reviews))
        np.random.shuffle(arr)
        for index in arr:
            review = training_reviews[index]
            user, item, rating = review.get_user(), review.get_item(), review.get_rating()
            text, theta = training_texts[:, index], training_thetas[:, index]
            b_u, gamma_u = self.users[user]
            b_i, gamma_i = self.items[item]

            delta_rating = self.overall_bias + b_u + b_i + np.dot(gamma_u, gamma_i) - rating
            delta_topics = self.k / len(text) * (text - np.dot(self.topics, theta))
            delta_unified = np.dot(theta, gamma_u) - rating

            self.update_topics_stochastic(delta_topics, theta)
            if delta_unified > 0:
                self.update_unified_profiles(user, item, delta_rating, b_u, b_i, gamma_u, gamma_i, delta_unified, theta)
                theta -= self.learning_rate * (
                    np.dot(self.topics.transpose(),
                           delta_topics) + self.balance * delta_unified * gamma_u + self.l2_weight * theta)
            else:
                # Update the latent profiles classically
                self.update_profiles(user, item, delta_rating, b_u, b_i, gamma_u, gamma_i)
                theta -= self.learning_rate * (np.dot(self.topics.transpose(), delta_topics) + self.l2_weight * theta)
            training_thetas[:, index] = theta

    def fit_with_validation(self, training_reviews, validation_reviews):
        """
        With the model using the explicitly given training and validation sets.
        :param training_reviews: training set.
        :param validation_reviews: validation set.
        :return: nothing, modifications are made in place.
        """
        OverallBias.fit(self, training_reviews)
        self.initialize(training_reviews)
        evaluation = RmseEvaluation(training_reviews, validation_reviews, None)
        training_texts = self.build_text_matrix(training_reviews)
        training_thetas = np.random.rand(self.k, len(training_reviews))
        if self.using_warm_restart:
            for epoch in range(10):
                TopicExtractor.update_thetas(self, training_texts, training_thetas)
        for epoch in range(1, self.epochs + 1):
            self.learning_rate = self.eta_0 / float(1 + epoch)
            self.one_unified_epoch(training_reviews, training_texts, training_thetas)
            training_rmse, validation_rmse, test_rmse = evaluation.evaluate(self)
            loss = self.compute_loss(training_texts, training_thetas)
            print("Perf @ epoch % {}: {} (training), {} (validation), {} (loss)".format(
                epoch, training_rmse, validation_rmse, loss))
