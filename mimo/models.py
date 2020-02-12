import copy

import numpy as np
import scipy.special as special
from scipy.special import logsumexp

from mimo.abstractions import ModelEM, ModelGibbsSampling, ModelMeanField
from mimo.abstractions import MeanField

from mimo.util.general import sample_discrete_from_log


#  internal labels class
class Labels:
    def __init__(self, model, data=None, N=None,
                 z=None, initialize_from_prior=True):

        assert data is not None or (N is not None and z is None)

        self.model = model

        if data is None:
            self._generate(N)
        else:
            self.data = data

            if z is not None:
                self.z = z
            elif initialize_from_prior:
                self._generate(len(data))
            else:
                self.resample()

    def _generate(self, N):
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
        if not hasattr(self, '_normalizer') or self._normalizer is None:
            scores = self._compute_scores()
            self._normalizer = logsumexp(scores[~np.isnan(self.data).any(1)], axis=1).sum()
        return self._normalizer

    def _compute_scores(self):
        data, K = self.data, len(self.components)
        scores = np.empty((data.shape[0], K))
        for idx, c in enumerate(self.components):
            scores[:, idx] = c.log_likelihood(data)
        scores += self.gating.log_likelihood(np.arange(K))
        scores[np.isnan(data).any(1)] = 0.  # missing data
        return scores

    def clear_caches(self):
        self._normalizer = None

    # Gibbs sampling
    def resample(self):
        scores = self._compute_scores()
        self.z, lognorms = sample_discrete_from_log(scores, axis=1, return_lognorms=True)
        self._normalizer = lognorms[~np.isnan(self.data).any(1)].sum()

    def copy_sample(self):
        new = copy.copy(self)
        new.z = self.z.copy()
        return new

    def get_responsibility(self, data=None, importance=None):
        data = self.data if data is None else data
        N, K = data.shape[0], len(self.components)

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N, K))
        for idx, c in enumerate(self.components):
            component_scores[:, idx] = c.expected_log_likelihood(data)
        component_scores = np.nan_to_num(component_scores)

        logpitilde = self.gating.expected_log_likelihood(np.arange(len(self.components)))
        logr = logpitilde + component_scores

        r = np.exp(logr - logr.max(1)[:, np.newaxis])
        r /= r.sum(1)[:, np.newaxis]

        if importance is not None:
            r = r * importance[:, np.newaxis]

        return r

    # Mean Field
    def meanfield_update(self, importance=None):
        # get responsibilities
        self.r = self.get_responsibility(importance=importance)
        # for plotting
        self.z = self.r.argmax(1)

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the mean field
        # variational lower bound
        errs = np.seterr(invalid='ignore', divide='ignore')
        prod = self.r * np.log(self.r)
        prod[np.isnan(prod)] = 0.  # 0 * -inf = 0.
        np.seterr(**errs)

        logpitilde = self.gating.expected_log_likelihood(np.arange(len(self.components)))

        q_entropy = - prod.sum()
        p_avgengy = (self.r * logpitilde).sum()

        return p_avgengy + q_entropy

    # EM
    def estep(self):
        data, N, K = self.data, self.data.shape[0], len(self.components)

        self.expectations = np.empty((N, K))
        for idx, c in enumerate(self.components):
            self.expectations[:, idx] = c.log_likelihood(data)
        self.expectations = np.nan_to_num(self.expectations)

        self.expectations += self.gating.log_likelihood(np.arange(K))

        self.expectations -= self.expectations.max(1)[:, np.newaxis]
        np.exp(self.expectations, out=self.expectations)
        self.expectations /= self.expectations.sum(1)[:, np.newaxis]

        self.z = self.expectations.argmax(1)


class Mixture(ModelEM, ModelGibbsSampling, ModelMeanField):
    """
    This class is for mixtures of other distributions.
    """
    _labels_class = Labels

    def __init__(self, gating, components):
        assert len(components) > 0

        self.gating = gating
        self.components = components

        self.labels_list = []

    def add_data(self, data, **kwargs):
        self.labels_list.append(self._labels_class(data=np.asarray(data), model=self, **kwargs))
        return self.labels_list[-1]

    def clear_data(self):
        self.labels_list = []

    @property
    def N(self):
        return len(self.components)

    def generate(self, N, keep=True, resp=False):
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

        if resp:
            r = templabels.get_responsibility()
            return out, templabels.z, r
        else:
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
    def resample_model(self, importance=[]):
        self.resample_components(importance=importance)
        self.resample_gating()
        self.resample_labels()

    def resample_gating(self):
        self.gating.resample([l.z for l in self.labels_list])
        self.clear_caches()

    def resample_components(self, importance=[]):
        for idx, c in enumerate(self.components):
            c.resample(data=[l.data[l.z == idx] for l in self.labels_list],
                       importance=[imp[l.z == idx] for (imp, l) in zip(importance, self.labels_list)])

    def resample_labels(self):
        for l in self.labels_list:
            l.resample()

    def copy_sample(self):
        new = copy.copy(self)
        new.components = [c.copy_sample() for c in self.components]
        new.weights = self.gating.copy_sample()
        new.labels_list = [l.copy_sample() for l in self.labels_list]
        for l in new.labels_list:
            l.model = new
        return new

    # Mean Field
    def meanfield_coordinate_descent_step(self, importance=[]):
        assert all(isinstance(c, MeanField) for c in self.components), 'Components must implement MeanField'
        assert len(self.labels_list) > 0, 'Must have data to run MeanField'

        self._meanfield_update_sweep(importance=importance)
        return self._vlb()

    def _meanfield_update_sweep(self, importance=[]):
        # NOTE: to interleave mean field steps with Gibbs sampling steps, label
        # updates need to come first, otherwise the sampled updates will be
        # ignored and the model will essentially stay where it was the last time
        # mean field updates were run
        # TODO fix that, seed with sample from variational distribution
        self.meanfield_update_labels(importance=importance)
        self.meanfield_update_parameters()

    def meanfield_update_labels(self, importance=[]):
        for idx, l in enumerate(self.labels_list):
            if not importance:
                l.meanfield_update(importance=None)
            else:
                l.meanfield_update(importance=importance[idx])

    def meanfield_update_parameters(self):
        self.meanfield_update_components()
        self.meanfield_update_gating()

    def meanfield_update_gating(self):
        self.gating.meanfield_update(None, [l.r for l in self.labels_list])
        self.clear_caches()

    def meanfield_update_components(self):
        for idx, c in enumerate(self.components):
            c.meanfield_update([l.data for l in self.labels_list], [l.r[:, idx] for l in self.labels_list])
        self.clear_caches()

    def _vlb(self):
        vlb = 0.
        vlb += sum(l.get_vlb() for l in self.labels_list)
        vlb += self.gating.get_vlb()
        vlb += sum(c.get_vlb() for c in self.components)
        for l in self.labels_list:
            vlb += np.sum([r.dot(c.expected_log_likelihood(l.data)) for c, r in zip(self.components, l.r.T)])

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
                                [l.r[:, idx] for l in mb_labels_list], prob, stepsize)

    def _meanfield_sgdstep_gating(self, mb_labels_list, prob, stepsize):
        self.gating.meanfield_sgdstep(None, [l.r for l in mb_labels_list], prob, stepsize)

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

    def predictive_likelihoods(self, test_data, forecast_horizons):
        likes = self._log_likelihoods(test_data)
        return [likes[k:] for k in forecast_horizons]

    def block_predictive_likelihoods(self, test_data, blocklens):
        csums = np.cumsum(self._log_likelihoods(test_data))
        outs = []
        for k in blocklens:
            outs.append(csums[k:] - csums[:-k])
        return outs
