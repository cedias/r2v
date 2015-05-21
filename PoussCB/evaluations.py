# coding=utf-8
import numpy as np


class RmseEvaluation:
    u"""
    Class used to compute RMSE (Root Mean Squared Error) of models.
    """
    def __init__(self, training_reviews, validation_reviews, test_reviews):
        u"""
        Sets a new evaluation campaign using given training, validation and test sets.
        Any set can be set as None, the evaluation for this set will then be None as well.
        :param training_reviews: the training set
        :param validation_reviews: the validation set
        :param test_reviews: the test set.
        :return: nothing ?
        """
        self.training_reviews = training_reviews
        self.validation_reviews = validation_reviews
        self.test_reviews = test_reviews

    def compute_rmse(self, recommender_system, reviews):
        u"""
        Computes the RMSE of given recommender system on given list of reviews.
        :param recommender_system: a recommender system (must have a predict(user, item) method)
        :param reviews: a list of reviews.
        :return: the RMSE.
        """
        rmse = None
        if reviews is not None:
            deltas = [recommender_system.predict(r.get_user(), r.get_item()) - r.get_rating() for r in reviews]
            squared_deltas = [x*x for x in deltas]
            rmse = np.sqrt(np.mean(squared_deltas))

        return rmse

    def evaluate(self, recommender_system):
        """
        Computes the RMSE of given recommender system on set training, validation and test sets.
        :param recommender_system: a recommender system (must have a predict(user, item) method)
        :return: training, validation and test RMSE. Some might be None if the corresponding set is None.
        """
        training_rmse = self.compute_rmse(recommender_system, self.training_reviews)
        validation_rmse = self.compute_rmse(recommender_system, self.validation_reviews)
        test_rmse = self.compute_rmse(recommender_system, self.test_reviews)
        return training_rmse, validation_rmse, test_rmse