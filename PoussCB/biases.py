# coding=utf-8
import numpy as np


class OverallBias:
    u"""
    Overall bias model: the prediction is constant w.r.t. users and items and is the mean rating estimated on training
    reviews.
    """
    def __init__(self):
        u"""
        Initializes the overall bias to NaN.
        :return:
        """
        # For now, the bias is not a number.
        self.overall_bias = np.nan

    def fit(self, training_reviews):
        u"""
        Estimates the overall bias (average rating) on given training reviews. It is not incremental, such subsequent
        calls each overrides the computations of previous calls.
        :param training_reviews: the training reviews to estimate the given rating from.
        :return: nothing.
        """
        self.overall_bias = np.mean([review.get_rating() for review in training_reviews])

    def predict(self, user, item):
        u"""
        Predict the rating that given user will give to give item.
        This method is defined as such so extending classes have a correct behavior: note that for the overall bias,
        the prediction is always the same and does not depend on the user nor the item.
        :param user: user index
        :param item: item index
        :return: the predicted rating.
        """
        return self.overall_bias


def _update_bias_profile(profiles, profile_id, rating):
    """
    Utility function that increments profiles dictionary. Shared by UserBias and ItemBias classes.
    :param profiles: the profiles as a dictionary.
    :param profile_id: the key to access the profile to update.
    :param rating: the rating to update the profile with.
    :return: nothing, the profiles are updated in place.
    """
    if profile_id not in profiles:
        profiles[profile_id] = {u"acc": rating, u"count": 1}
    else:
        profile = profiles[profile_id]
        profile[u"acc"] += rating
        profile[u"count"] += 1


def _compute_bias(profiles):
    """
    Helper function that computes the bias of each profile given the accumulator and the count:
        bias = acc / count
    :param profiles: the profiles as a dictionary.
    :return: nothing, the profiles are updated in place.
    """
    for profile_id, profile in profiles.items():
        profile[u"bias"] = profile[u"acc"] / float(profile[u"count"])


class UserBias(OverallBias):
    u"""
    This is the user bias model: it computes the average rating per user on the training set. Then the prediction
    is always the same for a user (its average rating).
    This class extends OverallBias both to share the fit/predict methods and to have a default prediction for unseen
    users.
    """
    def __init__(self):
        """
        Initializes the overall bias to NaN and the user profiles to an empty dictionary.
        :return: nothing
        """
        OverallBias.__init__(self)
        self.users = {}

    def fit(self, training_reviews):
        u"""
        Estimates the bias for users and the overall bias (average rating) on given training reviews.
        It is not incremental, such subsequent calls each overrides the computations of previous calls.
        :param training_reviews: the training reviews to estimate the given rating from.
        :return: nothing.
        """
        # Fit the overall bias
        OverallBias.fit(self, training_reviews)
        for review in training_reviews:
            user = review.get_user()
            _update_bias_profile(self.users, user, review.get_rating())
        _compute_bias(self.users)

    def predict(self, user, item):
        """
        Predicts the rating that given user will give to given item.
        For this model the item is not relevant, only the user is.
        :param user: user index
        :param item: item index
        :return: the average rating of given user on the training set, if any, else the overall bias.
        """
        if user not in self.users:
            return OverallBias.predict(self, user, item)
        return self.users[user][u"bias"]


class ItemBias(OverallBias):
    u"""
    This is the item bias model: it computes the average rating per item on the training set. Then the prediction
    is always the same for an item (its average rating).
    This class extends OverallBias both to share the fit/predict methods and to have a default prediction for unseen
    items.
    """
    def __init__(self):
        """
        Initializes the overall bias to NaN and the item profiles to an empty dictionary.
        :return: nothing
        """
        OverallBias.__init__(self)
        self.items = {}

    def fit(self, training_reviews):
        u"""
        Estimates the bias for items and the overall bias (average rating) on given training reviews.
        It is not incremental, such subsequent calls each overrides the computations of previous calls.
        :param training_reviews: the training reviews to estimate the given rating from.
        :return: nothing.
        """
        # Fit the overall bias
        OverallBias.fit(self, training_reviews)
        for review in training_reviews:
            item = review.get_item()
            _update_bias_profile(self.items, item, review.get_rating())
        _compute_bias(self.items)

    def predict(self, user, item):
        """
        Predicts the rating that given user will give to given item.
        For this model the user is not relevant, only the item is.
        :param user: user index
        :param item: item index
        :return: the average rating of given item on the training set, if any, else the overall bias.
        """
        if item not in self.items:
            return OverallBias.predict(self, user, item)
        return self.items[item][u"bias"]


