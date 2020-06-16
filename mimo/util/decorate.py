def pass_obs_arg(f):
    def wrapper(self, obs=None, **kwargs):
        if obs is None:
            assert self.has_data()
            obs = [_obs for _obs in self.obs]
        else:
            obs = obs if isinstance(obs, list) else [obs]

        return f(self, obs, **kwargs)
    return wrapper


def pass_obs_and_labels_arg(f):
    def wrapper(self, obs=None, labels=None, **kwargs):
        if obs is None or labels is None:
            assert self.has_data()
            obs = [_obs for _obs in self.obs]
            labels = self.labels
        else:
            obs = obs if isinstance(obs, list) else [obs]
            labels = [self.gating.likelihood.rvs(len(_obs)) for _obs in obs]\
                if labels is None else labels

        return f(self, obs, labels, **kwargs)
    return wrapper


def pass_target_and_input_arg(f):
    def wrapper(self, y=None, x=None, **kwargs):
        if y is None or x is None:
            assert self.has_data()
            y = [_y for _y in self.target]
            x = [_x for _x in self.input]
        else:
            y = y if isinstance(y, list) else [y]
            x = x if isinstance(x, list) else [x]

        return f(self, y, x, **kwargs)
    return wrapper


def pass_target_input_and_labels_arg(f):
    def wrapper(self, y=None, x=None, z=None, **kwargs):
        if y is None or x is None and z is None:
            assert self.has_data()
            y = [_y for _y in self.target]
            x = [_x for _x in self.input]
            z = self.labels
        else:
            y = y if isinstance(y, list) else [y]
            x = x if isinstance(x, list) else [x]
            z = [self.gating.likelihood.rvs(len(_y)) for _y in y]\
                if z is None else z

        return f(self, y, x, z, **kwargs)
    return wrapper