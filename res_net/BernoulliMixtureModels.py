# author: Nikola Zubic

from scipy.special import logsumexp
import numpy as np
from sklearn.model_selection import train_test_split


class BernoulliMixtureModel(object):
    """
    The presence of proteins of interest is given by a binary value: 1 if present, 0 if not present.
    Because we want to search for clusters of these discrete values, we 'll use a discrete clustering algorithm - BMM.
    """
    def __init__(self, number_of_components, maximum_number_of_iterations, tolerance=1e-3):
        """
        For each sample x_n of our N data spots there is a latent variable z_n that holds 1 for the component k that
        generated x_n and 0 for all others.
        Each target protein x_d is binary. If we would like to describe the distribution of a specific organelle, we
        would flip a coin. For each of the D = 28 target proteins we could flip a coin, with zero on one side and one
        on the other side. If these targets are independent within one component k, then we can observe current target
        protein patterns. So, these protein patterns (organelles) that together have frequent 1's are those which
        represent one target group.

        For example, if we have lysosomes and endosomes, and we know that they belong to group component k = 2, then
        the probability to observe them within this group can be calculated and we get mu_2,lysosome=mu_2,lysosome=0.99.
        On the other side probability for some other organelle 0.3, and for all the others it is 0.02.
        So, if we have a sample that fits to that target combination, it would produce high probability value p, and
        we can say that it can belong to that group. We use Multinomial distribution (one true component for each
        sample).

        :param number_of_components: we have K components, where every component represents one target group
        :param maximum_number_of_iterations:
        :param tolerance:
        """
        self.number_of_components = number_of_components
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.tolerance = tolerance

    def __init__params(self):
        self.number_of_samples = self.x.shape[0]
        self.number_of_features = self.x.shape[1]

        self.pi = 1 / self.number_of_components * np.ones(self.number_of_components)
        self.mu = np.random.RandomState(seed=0).uniform(low=0.25, high=0.75, size=(self.number_of_components,
                                                                                   self.number_of_features))
        self.normalize_mu()

    def remember_params(self):
        self.old_mu = self.mu.copy()
        self.old_pi = self.pi.copy()
        self.old_gamma = self.gamma.copy()

    def reset_params(self):
        self.mu = self.old_mu.copy()
        self.pi = self.old_pi.copy()
        self.gamma = self.old_gamma.copy()
        self.get_Neff()
        log_bernoullis = self.log_bernoulli(self.x)
        self.log_likelihood = self.get_log_likelihood(log_bernoullis)

    def normalize_mu(self):
        sum_over_features = np.sum(self.mu, axis=1)
        for k in range(self.number_of_components):
            self.mu[k,:] /= sum_over_features[k]

    def save_single_data_spot(self, x, mu):
        mu_place = np.where(np.max(mu, axis=0) <= 1e-15, 1e-15, mu)
        return np.tensordot(x, np.log(mu_place), (1, 1))

    def log_bernoulli(self, x):
        log_bernoullis = self.save_single_data_spot(x, self.mu)
        log_bernoullis += self.save_single_data_spot(1 - x, 1 - self.mu)
        return log_bernoullis

    def get_sample_log_likelihood(self, log_bernoullis):
        return logsumexp(np.log(self.pi[None, :]) + log_bernoullis, axis=1)

    def get_log_likelihood(self, log_bernoullis):
        return np.mean(self.get_sample_log_likelihood(log_bernoullis))

    def score(self, x):
        log_bernoullis = self.log_bernoulli(x)
        return self.get_log_likelihood(log_bernoullis)

    def score_samples(self, x):
        log_bernoullis = self.log_bernoulli(x)
        return self.get_sample_log_likelihood(log_bernoullis)

    def get_Neff(self):
        self.Neff = np.sum(self.gamma, axis=0)

    def get_mu(self):
        self.mu = np.einsum('ik,id -> kd', self.gamma, self.x) / self.Neff[:, None]

    def get_pi(self):
        self.pi = self.Neff / self.number_of_samples

    def get_responsibilities(self, log_bernoullis):
        gamma = np.zeros(shape=(log_bernoullis.shape[0], self.number_of_components))
        Z = logsumexp(np.log(self.pi[None, :]) + log_bernoullis, axis=1)
        for k in range(self.number_of_components):
            gamma[:, k] = np.exp(np.log(self.pi[k]) + log_bernoullis[:, k] - Z)
        return gamma

    def fit(self, x):
        self.x = x

        self.__init__params()

        log_bernoullis = self.log_bernoulli(self.x)

        self.old_log_likelihood = self.get_log_likelihood(log_bernoullis=log_bernoullis)

        for step in range(self.maximum_number_of_iterations):
            if step > 0:
                self.old_log_likelihood = self.log_likelihood

            # E-Step
            self.gamma = self.get_responsibilities(log_bernoullis)
            self.remember_params()

            # M-Step
            self.get_Neff()
            self.get_mu()
            self.get_pi()

            # Compute new log_likelihood:
            log_bernoullis = self.log_bernoulli(self.x)
            self.log_likelihood = self.get_log_likelihood(log_bernoullis)

            if np.isnan(self.log_likelihood):
                self.reset_params()
                print(self.log_likelihood)
                break
