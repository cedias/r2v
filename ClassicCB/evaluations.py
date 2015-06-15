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
            rmse = np.mean(squared_deltas)

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

class ErrorEvaluation:

    def __init__(self, training_reviews, validation_reviews, test_reviews,min_star=0,max_star=5,step=0.25):

        """
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
        self.min_star = min_star
        self.max_star = max_star
        self.step = step
        self.err_bins = list(np.arange(min_star,max_star,step)**2)



    def compute_eval(self, recommender_system, reviews):
        """
        Computes the Eval of given recommender system on given list of reviews.
        :param recommender_system: a recommender system (must have a predict(user, item) method)
        :param reviews: a list of reviews.
        :return: the RMSE.
        """

        if reviews is not None:

            err_cpt = list(np.zeros(len(self.err_bins)+1))
            deltas = [recommender_system.predict(r.get_user(), r.get_item()) - r.get_rating() for r in reviews]
            deltas = [x * x for x in deltas]
            for err in deltas:
                for i,val in enumerate(self.err_bins):
                    if err <= val:
                        err_cpt[i] += 1
                        break;
                    else:
                        if i == len(self.err_bins) - 1:
                            err_cpt[i+1] += 1

            err = (np.array(err_cpt)/(len(deltas)+.0))*100
            return err

    def evaluate(self, recommender_system):
        """
        Computes the RMSE of given recommender system on set training, validation and test sets.
        :param recommender_system: a recommender system (must have a predict(user, item) method)
        :return: training, validation and test RMSE. Some might be None if the corresponding set is None.
        """
        training_rmse = self.compute_eval(recommender_system, self.training_reviews)
        validation_rmse = self.compute_eval(recommender_system, self.validation_reviews)
        test_rmse = self.compute_eval(recommender_system, self.test_reviews)
        return training_rmse, validation_rmse, test_rmse
