# coding=utf-8
import numpy as np


class TopicExtractor:
    u"""
    Topic extractor as a matrix factorization. The text matrix is explained as a product of factors: a dictionary
    and a topic distribution matrix.
    """
    def __init__(self, dictionary, k=5, epochs=100):
        u"""
        Sets the training parameters for the topic extractor.
        :param dictionary: a gensim dictionary to treat the texts (texts -> matrix).
        :param k: number of topics to extract.
        :param epochs: number of training epochs.
        :return: nothing ?
        """
        self.dictionary = dictionary
        self.k = k
        self.epochs = epochs
        self.topics = np.zeros((len(dictionary), k))

    def build_text_matrix(self, training_reviews):
        u"""
        Computes the text matrix from given training text reviews.
        :param training_reviews: a list of text reviews.
        :return: the text matrix
        """
        text_matrix = np.zeros((len(self.dictionary), len(training_reviews)))
        for column, text_review in enumerate(training_reviews):
            for row, count in self.dictionary.doc2bow(text_review.get_text()):
                text_matrix[row, column] = 1
        return text_matrix

    def initialize(self):
        u"""
        Initializes the topic matrix.
        :return: nothing, the initialization is done in place.
        """
        # Use of the * to map tuple of ints to int, int
        self.topics = np.random.rand(*self.topics.shape)

    def update_thetas(self, text_matrix, thetas):
        u"""
        Updates the topic distribution matrix using multiplicative update rules (might not be optimal ... ).
        :param text_matrix: the text matrix.
        :param thetas: current topic distribution matrix.
        :return: nothing, the update is done in place.
        """
        thetas *= self.topics.transpose().dot(text_matrix) / (
            1e-9 + self.topics.transpose().dot(self.topics).dot(thetas))

    def update_topics(self, text_matrix, thetas):
        u"""
        Updates the topic matrix using multiplicative update rules (might not be optimal ... ).
        :param text_matrix: the text matrix.
        :param thetas: current topic distribution matrix.
        :return: nothing, the update is done in place/
        """
        self.topics *= text_matrix.dot(thetas.transpose()) / (1e-9 + self.topics.dot(thetas.dot(thetas.transpose())))

    def compute_loss(self, text_matrix, thetas):
        u"""
        Computes the loss (Frobenius) w.r.t. to given text matrix, topic distribution and current topic matrix.
        :param text_matrix: text matrix.
        :param thetas: topic distribution matrix.
        :return: the value of the loss (Frobenius, averaged).
        """
        return np.linalg.norm(text_matrix - np.dot(self.topics, thetas), ord=u"fro") / float(thetas.shape[1])

    def fit_transform(self, training_reviews):
        u"""
        Trains the model using set parameters and given training reviews.
        :param training_reviews: training reviews to use.
        :return: the topic distribution matrix.
        """
        text_matrix = self.build_text_matrix(training_reviews)
        self.initialize()
        thetas = np.random.rand(self.k, len(training_reviews))
        for epoch in range(self.epochs):
            self.update_thetas(text_matrix, thetas)
            self.update_topics(text_matrix, thetas)
            print(u"Loss @ epoch % 4d: %.5f" % (epoch, (self.compute_loss(text_matrix, thetas))))
        return thetas