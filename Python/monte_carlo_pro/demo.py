import importlib
import numpy as np

# try to import compiled backend (built with CMake and pybind11)
_backend = None
try:
    _backend = importlib.import_module('monte_carlo_pro._backend')
except Exception:
    try:
        _backend = importlib.import_module('_backend')
    except Exception:
        _backend = None


class MonteCarloPro:
    def __init__(self, backend='cpu', paths=100000):
        if backend not in ('cpu', 'gpu'):
            raise ValueError('backend must be "cpu" or "gpu"')
        self.backend = backend
        self.paths = int(paths)

    def price_option(self, S0, K, T, r, sigma, option_type='call'):
        opt_map = {
            'call': 'EuropeanCall',
            'put': 'EuropeanPut',
            'asian_call': 'AsianCall',
            'asian_put': 'AsianPut'
        }
        key = option_type if option_type in opt_map else option_type.lower()
        if key not in opt_map:
            raise ValueError('option_type must be one of: ' + ','.join(opt_map.keys()))

        if self.backend == 'cpu':
            if _backend is None:
                raise RuntimeError("CPU backend requires compiled C++ extension.")
            Opt = _backend.Option
            OptionType = _backend.OptionType
            opt = Opt()
            opt.type = OptionType.EuropeanCall if key == "call" else OptionType.EuropeanPut
            opt.S0, opt.K, opt.T, opt.r, opt.sigma = float(S0), float(K), float(T), float(r), float(sigma)
            price, stderr = _backend.price_cpu(opt, int(self.paths))
            return float(price), float(stderr)
        else:
            if _backend is None:
                raise RuntimeError('GPU backend not available. Build the C++/CUDA extension and ensure the module is importable.')
            # compose Option object defined in the extension
            Opt = _backend.Option
            OptionType = _backend.OptionType
            opt = Opt()
            # map
            if opt_map[key] == 'EuropeanCall':
                opt.type = OptionType.EuropeanCall
            elif opt_map[key] == 'EuropeanPut':
                opt.type = OptionType.EuropeanPut
            elif opt_map[key] == 'AsianCall':
                opt.type = OptionType.AsianCall
            else:
                opt.type = OptionType.AsianPut
            opt.S0 = float(S0)
            opt.K = float(K)
            opt.T = float(T)
            opt.r = float(r)
            opt.sigma = float(sigma)

            price, stderr = _backend.price_gpu(opt, int(self.paths))
            return float(price), float(stderr)
