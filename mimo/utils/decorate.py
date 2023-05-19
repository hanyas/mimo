def pass_obs_arg(f):
    def wrapper(self, obs=None, **kwargs):
        if obs is None:
            assert self.has_data()
            obs = self.obs

        return f(self, obs, **kwargs)
    return wrapper


def pass_obs_and_labels_arg(f):
    def wrapper(self, obs=None, labels=None, **kwargs):
        if obs is None and labels is None:
            assert self.has_data()
            obs = self.obs
            labels = self.labels
        elif obs is not None and labels is None:
            labels = self.gating.likelihood.rvs(len(obs))

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
            z = [self.gating.likelihood.rvs(len(_y)) for _y in y]

        return f(self, y, x, z, **kwargs)
    return wrapper
