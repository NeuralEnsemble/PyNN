class NativeRNG():
    """    
    Signals that the random numbers will be drawn by NEST's own RNGs and 
    takes care of transforming pyNN parameters for the random distributions
    to NEST parameters.
    """
    translations = {
        'binomial':       {'n': 'n', 'p': 'p'},
        'gamma':          {'theta': 'scale', 'k': 'order'},
        'exponential':    {'beta': 'lambda'},
        'lognormal':      {'mu': 'mu', 'sigma': 'sigma'},
        'normal':         {'mu': 'mu', 'sigma': 'sigma'},
        'normal_clipped': {'mu': 'mu', 'sigma': 'sigma', 'low': 'low', 'high': 'high'},
        'normal_clipped_to_boundary':
                          {'mu': 'mu', 'sigma': 'sigma', 'low': 'low', 'high': 'high'},
        'poisson':        {'lambda': 'lambda'},
        'uniform':        {'low': 'low', 'high': 'high'},
        'uniform_int':    {'low': 'low', 'high': 'high'},
        'vonmises':       {'mu': 'mu', 'kappa': 'kappa'},
    }
    def __init__(self,pynnDistribution):
        parameter_map = self.translations[pynnDistribution.name]
        self.parameters = dict((parameter_map[k], v) for k, v in pynnDistribution.parameters.items())
        self.parameters['distribution'] = pynnDistribution.name
        
