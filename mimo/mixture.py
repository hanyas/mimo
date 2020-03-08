import copy

import numpy as np
import scipy.special as special
from scipy.special import logsumexp

from mimo.abstractions import ModelEM, ModelGibbsSampling, ModelMeanField
from mimo.abstractions import MeanField

from mimo.util.general import sample_discrete_from_log

from mimo.distributions.bayesian import BayesianCategoricalWithDirichlet
from mimo.distributions.bayesian import BayesianCategoricalWithStickBreaking

import joblib
from joblib import Parallel, delayed
nb_cores = joblib.parallel.cpu_count()


# labels class assuming a dirichlet prior
class LabelsWithDirichlet:
    def __init__(self, model, data=None, N=None,
                 z=None, initialize_from_prior=True):

        assert data is not None or (N is not None and z is None)

        self.model = model

        self.normalizer = None
        self.expectations = None
        self.resps = None

        if data is None:
            self.generate(N)
        else:
            self.data = data

            if z is not None:
                self.z = z
            elif initialize_from_prior:
                self.generate(len(data))
            else:
                self.resample()

    def generate(self, N):
        self.z = self.gating.rvs(N)

    @property
    def N(self):
        return len(self.z)

    @property
    def gating(self):
        return self.model.gating

    @property
    def components(self):
        return self.model.components

    def log_likelihood(self):
        if not hasattr(self, 'normalizer') or self.normalizer is None:
            scores = self.compute_scores()
            self.normalizer = np.sum(logsumexp(scores[~np.isnan(self.data).any(1)], axis=1))
        return self.normalizer

    def clear_caches(self):
        self.normalizer = None

    # Gibbs sampling
    def resample(self):
        scores = self.compute_scores()
        self.z, lognorms = sample_discrete_from_log(scores, axis=1, return_lognorms=True)
        self.normalizer = np.sum(lognorms[~np.isnan(self.data).any(1)])

    def copy_sample(self):
        new = copy.copy(self)
        new.z = self.z.copy()
        return new

    def compute_scores(self, data=None):
        # compute responsibilities
        data = self.data if data is None else data
        N, K = data.shape[0], len(self.components)

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, c in enumerate(self.components):
            component_scores[:, idx] = c.log_likelihood(data)
        component_scores = np.nan_to_num(component_scores)

        logpitilde = self.gating.log_likelihood(np.arange(K))
        score = logpitilde + component_scores
        return score

    def compute_responsibilities(self, data=None):
        # compute responsibilities
        data = self.data if data is None else data
        N, K = data.shape[0], len(self.components)

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, c in enumerate(self.components):
            component_scores[:, idx] = c.expected_log_likelihood(data)
        component_scores = np.nan_to_num(component_scores)

        logpitilde = self.gating.expected_log_likelihood(np.arange(K))
        logr = logpitilde + component_scores

        r = np.exp(logr - np.max(logr, axis=1, keepdims=True))
        r /= np.sum(r, axis=1, keepdims=True)

        return r, logr

    # Mean Field
    def meanfield_update(self):
        # get responsibilities
        self.resps, _ = self.compute_responsibilities()
        # for plotting
        self.z = np.argmax(self.resps, axis=1)

    def variational_lowerbound(self):
        K = len(self.components)

        # return avg energy plus entropy
        errs = np.seterr(invalid='ignore', divide='ignore')
        prod = self.resps * np.log(self.resps)
        prod[np.isnan(prod)] = 0.  # 0 * -inf = 0.
        np.seterr(**errs)

        logpitilde = self.gating.expected_log_likelihood(np.arange(K))

        q_entropy = - prod.sum()
        p_avgengy = (self.resps * logpitilde).sum()

        return p_avgengy + q_entropy

    # EM
    def estep(self):
        data, N, K = self.data, self.data.shape[0], len(self.components)

        self.expectations = np.empty((N, K))
        for idx, c in enumerate(self.components):
            self.expectations[:, idx] = c.log_likelihood(data)
        self.expectations = np.nan_to_num(self.expectations)

        self.expectations += self.gating.log_likelihood(np.arange(K))

        self.expectations -= np.max(self.expectations, axis=1, keepdims=True)
        np.exp(self.expectations, out=self.expectations)
        self.expectations /= np.sum(self.expectations, axis=1, keepdims=True)

        self.z = np.argmax(self.expectations, axis=1)


# labels class assuming a stick-breaking prior
class LabelsWithStickBreaking:
    def __init__(self, model, data=None, N=None,
                 z=None, initialize_from_prior=True):

        assert data is not None or (N is not None and z is None)

        self.model = model

        self.normalizer = None
        self.expectations = None
        self.resps = None

        if data is None:
            self.generate(N)
        else:
            self.data = data

            if z is not None:
                self.z = z
            elif initialize_from_prior:
                self.generate(len(data))
            else:
                self.resample()

    def generate(self, N):
        self.z = self.gating.rvs(N)

    @property
    def N(self):
        return len(self.z)

    @property
    def gating(self):
        return self.model.gating

    @property
    def components(self):
        return self.model.components

    def log_likelihood(self):
        if not hasattr(self, 'normalizer') or self.normalizer is None:
            scores = self.compute_scores()
            self.normalizer = np.sum(logsumexp(scores[~np.isnan(self.data).any(1)], axis=1))
        return self.normalizer

    def clear_caches(self):
        self.normalizer = None

    # Gibbs sampling
    def resample(self):
        # TODO Are we doing this the correct way?
        scores = self.compute_scores()
        self.z, lognorms = sample_discrete_from_log(scores, axis=1, return_lognorms=True)
        self.normalizer = np.sum(lognorms[~np.isnan(self.data).any(1)])

    def copy_sample(self):
        new = copy.copy(self)
        new.z = self.z.copy()
        return new

    def compute_scores(self, data=None):
        data = self.data if data is None else data
        N, K = data.shape[0], len(self.components)

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, c in enumerate(self.components):
            component_scores[:, idx] = c.log_likelihood(data)
        component_scores = np.nan_to_num(component_scores)

        gating_scores = self.gating.log_likelihood(np.arange(K))

        score = gating_scores + component_scores
        return score

    def compute_responsibilities(self, data=None):
        data = self.data if data is None else data
        N, K = data.shape[0], len(self.components)

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, c in enumerate(self.components):
            component_scores[:, idx] = c.expected_log_likelihood(data)
        component_scores = np.nan_to_num(component_scores)

        E_log_stick, E_log_rest = self.gating.expected_log_likelihood(np.arange(K))
        gating_scores = np.take(E_log_stick + np.hstack((0, np.cumsum(E_log_rest)[:-1])), np.arange(K))

        logr = gating_scores + component_scores

        r = np.exp(logr - np.max(logr, axis=1, keepdims=True))
        r /= np.sum(r, axis=1, keepdims=True)

        return r, logr

    # Mean Field
    def meanfield_update(self):
        # get responsibilities
        self.resps, _ = self.compute_responsibilities()
        # for plotting
        self.z = np.argmax(self.resps, axis=1)

    def variational_lowerbound(self):
        K = len(self.components)

        # return avg energy plus entropy
        errs = np.seterr(invalid='ignore', divide='ignore')
        prod = self.resps * np.log(self.resps)
        prod[np.isnan(prod)] = 0.  # 0 * -inf = 0.
        np.seterr(**errs)

        q_entropy = - prod.sum()

        counts = self.resps
        cumcounts = np.hstack((np.cumsum(counts[:, ::-1], axis=1)[:, -2::-1],
                               np.zeros((len(counts), 1))))

        E_log_stick, E_log_rest = self.gating.expected_log_likelihood(np.arange(K))
        p_avgengy = np.sum(cumcounts * E_log_rest + counts * E_log_stick)

        return p_avgengy + q_entropy

    # EM
    def estep(self):
        raise NotImplementedError


class Mixture(ModelEM, ModelGibbsSampling, ModelMeanField):
    """
    This class is for mixtures of other distributions.
    """

    def __init__(self, gating, components):
        assert len(components) > 0

        self.gating = gating
        self.components = components

        self._labels_class = None
        if isinstance(self.gating, BayesianCategoricalWithDirichlet):
            self._labels_class = LabelsWithDirichlet
        elif isinstance(self.gating, BayesianCategoricalWithStickBreaking):
            self._labels_class = LabelsWithStickBreaking
        else:
            raise NotImplementedError

        self.labels_list = []

    def add_data(self, data, **kwargs):
        self.labels_list.append(self._labels_class(data=np.asarray(data), model=self, **kwargs))
        return self.labels_list[-1]

    def clear_data(self):
        self.labels_list = []

    @property
    def N(self):
        return len(self.components)

    def generate(self, N, keep=True):
        templabels = self._labels_class(model=self, N=N)

        out = np.empty(self.components[0].rvs(N).shape)
        counts = np.bincount(templabels.z, minlength=self.N)
        for idx, (c, count) in enumerate(zip(self.components, counts)):
            out[templabels.z == idx, ...] = c.rvs(count)

        perm = np.random.permutation(N)
        out = out[perm]
        templabels.z = templabels.z[perm]

        if keep:
            templabels.data = out
            self.labels_list.append(templabels)

        return out, templabels.z

    def clear_caches(self):
        for l in self.labels_list:
            l.clear_caches()

    def _log_likelihoods(self, x):
        # NOTE: nans propagate as nans
        x = np.asarray(x)
        K = len(self.components)
        vals = np.empty((x.shape[0], K))
        for idx, c in enumerate(self.components):
            vals[:, idx] = c.log_likelihood(x)
        vals += self.gating.log_likelihood(np.arange(K))
        return logsumexp(vals, axis=1)

    def log_likelihood(self, x=None):
        if x is None:
            return sum(l.log_likelihood() for l in self.labels_list)
        else:
            assert isinstance(x, (np.ndarray, list))
            if isinstance(x, list):
                return sum(self.log_likelihood(d) for d in x)
            else:
                self.add_data(x)
                return self.labels_list.pop().log_likelihood()

    # Gibbs sampling
    def resample_model(self, num_procs=0):
        self.resample_components(num_procs=num_procs)
        self.resample_gating()
        self.resample_labels(num_procs=num_procs)

    def resample_gating(self):
        self.gating.resample([l.z for l in self.labels_list])
        self.clear_caches()

    def resample_components(self, num_procs=0):
        if num_procs == 0:
            for idx, c in enumerate(self.components):
                c.resample(data=[l.data[l.z == idx] for l in self.labels_list])
        else:
            self._resample_components_joblib(num_procs)
        self.clear_caches()

    def resample_labels(self, num_procs=0):
        if num_procs == 0:
            for l in self.labels_list:
                l.resample()
        else:
            self._resample_labels_joblib(num_procs)

    def _resample_components_joblib(self, num_procs):
        from . import parallel_mixture

        parallel_mixture.model = self
        parallel_mixture.labels_list = self.labels_list

        if len(self.components) > 0:
            params = Parallel(n_jobs=num_procs, backend='threading')\
                    (delayed(parallel_mixture._get_sampled_component_params)(idx)
                     for idx in range(len(self.components)))

        for c, p in zip(self.components, params):
            c.parameters = p

    def _resample_labels_joblib(self, num_procs):
        from . import parallel_mixture

        if len(self.labels_list) > 0:
            parallel_mixture.model = self

            raw = Parallel(n_jobs=num_procs, backend='threading')\
                (delayed(parallel_mixture._get_sampled_labels)(idx)
                 for idx in range(len(self.labels_list)))

            for l, (z, normalizer) in zip(self.labels_list, raw):
                l.z, l._normalizer = z, normalizer

    def copy_sample(self):
        new = copy.copy(self)
        new.components = [c.copy_sample() for c in self.components]
        new.weights = self.gating.copy_sample()
        new.labels_list = [l.copy_sample() for l in self.labels_list]
        for l in new.labels_list:
            l.model = new
        return new

    # Mean Field
    def meanfield_coordinate_descent_step(self):
        assert all(isinstance(c, MeanField) for c in self.components), 'Components must implement MeanField'
        assert len(self.labels_list) > 0, 'Must have data to run MeanField'

        self._meanfield_update_sweep()
        return self.variational_lowerbound()

    def _meanfield_update_sweep(self):
        # NOTE: to interleave mean field steps with Gibbs sampling steps, label
        # updates need to come first, otherwise the sampled updates will be
        # ignored and the model will essentially stay where it was the last time
        # mean field updates were run
        # TODO fix that, seed with sample from variational distribution
        self.meanfield_update_labels()
        self.meanfield_update_parameters()

    def meanfield_update_labels(self):
        for idx, l in enumerate(self.labels_list):
            l.meanfield_update()

    def meanfield_update_parameters(self):
        self.meanfield_update_components()
        self.meanfield_update_gating()

    def meanfield_update_gating(self):
        self.gating.meanfield_update(None, [l.resps for l in self.labels_list])
        self.clear_caches()

    def meanfield_update_components(self):
        for idx, c in enumerate(self.components):
            c.meanfield_update([l.data for l in self.labels_list], [l.resps[:, idx] for l in self.labels_list])
        self.clear_caches()

    def variational_lowerbound(self):
        vlb = 0.
        vlb += sum(l.variational_lowerbound() for l in self.labels_list)
        vlb += self.gating.variational_lowerbound()
        vlb += sum(c.variational_lowerbound() for c in self.components)
        for l in self.labels_list:
            vlb += np.sum([r.dot(c.expected_log_likelihood(l.data)) for c, r in zip(self.components, l.resps.T)])

        # add in symmetry factor (if we're actually symmetric)
        if len(set(type(c) for c in self.components)) == 1:
            vlb += special.gammaln(len(self.components) + 1)

        return vlb

    # SVI
    def meanfield_sgdstep(self, minibatch, prob, stepsize, **kwargs):
        minibatch = minibatch if isinstance(minibatch, list) else [minibatch]
        mb_labels_list = []
        for data in minibatch:
            self.add_data(data, z=np.empty(data.shape[0]), **kwargs)  # NOTE: dummy
            mb_labels_list.append(self.labels_list.pop())

        for l in mb_labels_list:
            l.meanfield_update()

        self._meanfield_sgdstep_parameters(mb_labels_list, prob, stepsize)

    def _meanfield_sgdstep_parameters(self, mb_labels_list, prob, stepsize):
        self._meanfield_sgdstep_components(mb_labels_list, prob, stepsize)
        self._meanfield_sgdstep_gating(mb_labels_list, prob, stepsize)

    def _meanfield_sgdstep_components(self, mb_labels_list, prob, stepsize):
        for idx, c in enumerate(self.components):
            c.meanfield_sgdstep([l.data for l in mb_labels_list],
                                [l.resps[:, idx] for l in mb_labels_list], prob, stepsize)

    def _meanfield_sgdstep_gating(self, mb_labels_list, prob, stepsize):
        self.gating.meanfield_sgdstep(None, [l.resps for l in mb_labels_list], prob, stepsize)

    # EM
    def em_step(self):
        # assert all(isinstance(c,MaxLikelihood) for c in self.components), \
        #         'Components must implement MaxLikelihood'
        assert len(self.labels_list) > 0, 'Must have data to run EM'

        # E step
        for l in self.labels_list:
            l.estep()

        # M step
        # component parameters
        for idx, c in enumerate(self.components):
            c.max_likelihood([l.data for l in self.labels_list], [l.expectations[:, idx] for l in self.labels_list])

        # mixture weights
        self.gating.max_likelihood(None, [l.expectations for l in self.labels_list])

    @property
    def num_parameters(self):
        return sum(c.num_parameters for c in self.components) + self.gating.num_parameters

    def bic(self, data=None):
        """
        BIC on the passed data.
        If passed data is None (default), calculates BIC on the model's assigned data.
        """
        # NOTE: in principle this method computes the BIC only after finding the
        # maximum likelihood parameters (or, of course, an EM fixed-point as an
        # approximation!)
        if data is None:
            assert len(self.labels_list) > 0, "No data available"
            return -2 * sum(self.log_likelihood(l.data) for l in self.labels_list)\
                   + self.num_parameters * np.log(sum(l.data.shape[0] for l in self.labels_list))
        else:
            return -2 * self.log_likelihood(data) + self.num_parameters * np.log(data.shape[0])

    def aic(self):
        # NOTE: in principle this method computes the AIC only after finding the
        # maximum likelihood parameters (or, of course, an EM fixed-point as an
        # approximation!)
        assert len(self.labels_list) > 0, 'Must have data to get AIC'
        return 2 * self.num_parameters - 2 * sum(self.log_likelihood(l.data) for l in self.labels_list)

    # Misc
    @property
    def used_labels(self):
        if len(self.labels_list) > 0:
            label_usages = sum(np.bincount(l.z, minlength=self.N) for l in self.labels_list)
            used_labels, = np.where(label_usages > 0)
        else:
            used_labels = np.argsort(self.gating.probs)[-1:-11:-1]
        return used_labels

    def plot(self, color=None, legend=False, alpha=None, update=False, draw=True):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        artists = []

        # get colors
        cmap = cm.get_cmap()
        if color is None:
            label_colors = dict((idx, cmap(v)) for idx, v in enumerate(np.linspace(0, 1, self.N, endpoint=True)))
        else:
            label_colors = dict((idx, color) for idx in range(self.N))

        # plot data scatter
        for l in self.labels_list:
            colorseq = [label_colors[label] for label in l.z]
            if update and hasattr(l, '_data_scatter'):
                l._data_scatter.set_offsets(l.data[:, :2])
                l._data_scatter.set_color(colorseq)
            else:
                l._data_scatter = plt.scatter(l.data[:, 0], l.data[:, 1], c=colorseq, s=5)
            artists.append(l._data_scatter)

        # plot parameters
        axis = plt.axis()
        for label, (c, w) in enumerate(zip(self.components, self.gating.probs)):
            artists.extend(c.plot(color=label_colors[label],
                                  label='%d' % label,
                                  alpha=min(0.25, 1. - (1. - w) ** 2) / 0.25 if alpha is None else alpha,
                                  update=update, draw=False))
        plt.axis(axis)

        # add legend
        if legend and color is None:
            plt.legend([plt.Rectangle((0, 0), 1, 1, fc=c)
                        for i, c in label_colors.items() if i in self.used_labels],
                       [i for i in label_colors if i in self.used_labels], loc='best', ncol=2)

        if draw:
            plt.draw()

        return artists

    def clear_plot(self):
        for l in self.labels_list:
            if hasattr(l, '_data_scatter') and l._data_scatter is not None:
                del l._data_scatter

        for c in self.components:
            if hasattr(c, '_parameterplot') and c._parameterplot is not None:
                del c._parameterplot
            if hasattr(c, '_scatterplot') and c._scatterplot is not None:
                del c._scatterplot
