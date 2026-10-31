
from ATLAS.utils import jit, jit_method

import jax.numpy as jnp
import numpy as np

from glob import glob

from tqdm_joblib import ParallelPbar
from joblib import delayed

from jug.engine.session import TimingSession
from enterprise.pulsar import Pulsar as E_Pulsar
import pint.logging
import logging
from dataclasses import dataclass, asdict
from typing import Optional

LINEAR_PARAMS = ('offset', 'f', 'dm', 'fd', 'jump')

MAX_JOBS = 8
LINEAR_PARAMS = ['offset','f','dm','fd','jump'] # Matt also said NE_SW, but I dunno what that is
TEMPO2_ALIASES = ('tempo2', 't2', 'libstempo')

def _backend_flags(psr):
    """Backend flags for an enterprise pulsar, tolerant of flagless TOAs.

    enterprise's ``backend_flags`` raises when the TOAs carry no flags at all,
    which is the case for the IPTA mock data challenge sets read through
    tempo2. Fall back to a single unnamed backend -- what the PINT reader
    yields for the very same files.
    """
    try:
        return list(psr.backend_flags)
    except (ValueError, AttributeError):
        return [''] * len(psr.toas)

def load_pulsars(par, tim, use_enterprise=True, timing_package='pint'):
    # par could be a list of par files, a directory string, or a single par file string
    if isinstance(par, list): # list of par file strings
        par_files = par
    elif isinstance(par, str): # Could be a directory string or a single par file string
        if par.endswith('.par'):
            par_files = [par] # single par file string
        else: # directory string
            par_files = sorted(glob(f'{par}/*.par'))
    else:
        raise ValueError('Unrecognized par file input')
    
    # tim could be a list of tim files, a directory string, or a single tim file string
    if isinstance(tim, list): # list of tim file strings
        tim_files = tim
    elif isinstance(tim, str): # Could be a directory string or a single tim file string
        if tim.endswith('.tim'):
            tim_files = [tim] # single tim file string
        else: # directory string
            tim_files = sorted(glob(f'{tim}/*.tim'))
    else:
        raise ValueError('Unrecognized tim file input')
    
    # Ensure that the pars have corresponding tim files
    assert len(par_files) == len(tim_files), 'Number of par files must match number of tim files'

    def foo(i):
        # strip the extension before splitting, so that datasets named plainly
        # ("J0030+0451.par") pair up as well as NANOGrav's "{PSR}_PINT_DMX.par"
        pname = par_files[i].split('/')[-1].rsplit('.', 1)[0].split('_')[0]
        tname = tim_files[i].split('/')[-1].rsplit('.', 1)[0].split('_')[0]
        assert pname == tname, f'Par file {par_files[i]} and tim file {tim_files[i]} do not match'

        psr = Pulsar(par_files[i], tim_files[i], use_enterprise=use_enterprise,
                     timing_package=timing_package)
        return psr
    
    psrs = ParallelPbar("Loading pulsars...")(n_jobs=MAX_JOBS)(
        delayed(foo)(i) for i in range(len(par_files))
    )

    return psrs

class Pulsar:
    def __init__(self, par, tim, use_enterprise=True, timing_package='pint'):
        self.par_file = par
        self.tim_file = tim
        self.timing_package = timing_package

        # Construct the pulsar using enterprise
        # when we can construct Mmat with JUG, we can use that entirely
        if use_enterprise:
            pint.logging.setup(level="ERROR")
            logging.getLogger('enterprise').setLevel(logging.ERROR)

            use_tempo2 = str(timing_package).lower() in TEMPO2_ALIASES
            if use_tempo2:
                # tempo2 reads TCB par files and every BINARY model (including
                # tempo2's own auto-dispatching "BINARY T2") natively, so it
                # loads datasets PINT refuses or silently mis-scales.
                psr = E_Pulsar(par, tim, timing_package='tempo2',
                               sort=False, drop_t2pulsar=False) # keeps psr.t2pulsar
            else:
                # Be explicit: enterprise picks tempo2 when libstempo is
                # importable, so omitting this silently ignores
                # timing_package='pint' and then fails on psr.model.
                psr = E_Pulsar(par, tim, timing_package='pint',
                               sort=False, drop_pintpsr=False) # keeps psr.model

            self.name = psr.name

            # Actual TOA stuffs
            self.toas = jnp.array(psr.toas)
            self.residuals = jnp.array(psr.residuals)
            self.toaerrs = jnp.array(psr.toaerrs)
            self.freqs = jnp.array(psr.freqs)
            self.backend_flags = _backend_flags(psr) # Can't store as jnp array

            # Position stuffs
            self.raj = jnp.double(psr._raj)
            self.decj = jnp.double(psr._decj)
            self.pos = jnp.array(psr.pos)

            # Timing model parameters
            self.fit_param_names = list(psr.fitpars)[1:] # Ignore the offset
            if use_tempo2:
                self.fit_param_values = jnp.array([psr.t2pulsar[k].val for k in self.fit_param_names], dtype=jnp.float64)
                self.fit_param_uncertainties = jnp.array([psr.t2pulsar[k].err for k in self.fit_param_names], dtype=jnp.float64)
            else:
                self.fit_param_values = jnp.array([psr.model[k].value for k in self.fit_param_names], dtype=jnp.float64)
                self.fit_param_uncertainties = jnp.array([psr.model[k].uncertainty_value for k in self.fit_param_names], dtype=jnp.float64)

            # Linearized design matrix
            self.Mmat = jnp.array(psr.Mmat)
            self.Mmat_labels = list(psr.fitpars)

            # Label which columns are exactly linear
            linear = []
            for l in self.Mmat_labels:
                state = any(l.lower().startswith(lp) for lp in LINEAR_PARAMS)
                linear.append(state)
            self.Mmat_is_linear = jnp.array(linear)
            
        else:
            s = TimingSession(par, tim)
            result = s.fit_parameters(verbose=False)

            self.name = s.params['PSR']

            # Actual TOA stuffs
            self.toas = jnp.array(result['bat_sec'])
            self.residuals = jnp.array(result['residuals_us']) * 1e-6
            self.toaerrs = jnp.array([t.error_us for t in s.toas_data]) * 1e-6
            self.freqs = jnp.array(result['freq_bary_mhz'])
            self.backend_flags = list([t.flags['f'] for t in s.toas_data]) # Can't store as jnp array

            # Pulsar Distances
            pdist = np.load('../datasets/NG15/pulsar_distances_15yr.npz')[self.name][:2]
            self.pdist = jnp.array([(float(x), float(y)) for x, y in [pdist]])[0]

            # Position stuffs
            self.raj = float(s.params['_raj_rad'])
            self.decj = float(s.params['_decj_rad'])
            # Convert RAJ and DECJ to Cartesian coordinates
            pos = [jnp.cos(self.raj) * jnp.cos(self.decj),
                   jnp.sin(self.raj) * jnp.cos(self.decj),
                   jnp.sin(self.decj)]
            self.pos = jnp.array(pos)

            # Timing model parameters
            self.fit_param_names = list(result['final_params'].keys())
            self.fit_param_values = jnp.array(list(result['final_params'].values()), dtype=jnp.float64)
            self.fit_param_uncertainties = jnp.array(list(result['uncertainties'].values()), dtype=jnp.float64)
                                                                 
            # Linearized design matrix
            self.Mmat = jnp.array(result['design_matrix'])
            self.Mmat_labels = result['design_matrix_labels']

            # Label which columns are exactly linear
            linear = []
            for l in self.Mmat_labels:
                state = any(l.startswith(lp) for lp in LINEAR_PARAMS)
                linear.append(state)

            self.Mmat_is_linear = jnp.array(linear)
            self.Mmat_linear = self.Mmat[:, self.Mmat_is_linear]
            self.Mmat_linear_labels = [l for l, is_lin in zip(self.Mmat_labels, self.Mmat_is_linear) if is_lin]


@dataclass
class PulsarDataLoader:
    """One pulsar's data.  Load:  PulsarDataLoader(**np.load(path))
                            Save:  psr.save(path)
    """

    # Basic info
    name: str

    # TOA data
    toas: np.ndarray                      # [s]
    residuals: np.ndarray                 # [s]
    toaerrs: np.ndarray                   # [s]
    freqs: np.ndarray                     # [MHz]
    backend_flags: np.ndarray             # str per TOA

    # Position
    pos: np.ndarray                       # unit 3-vector

    # Linearized design matrix
    Mmat: np.ndarray
    Mmat_labels: list

    # ---- everything below is optional ---------------------------------
    par_file: Optional[str] = None
    tim_file: Optional[str] = None
    timing_package: Optional[str] = None
    raj: Optional[float] = None           # [rad]; computed from pos if missing
    decj: Optional[float] = None
    pdist: Optional[np.ndarray] = None    # (mean, std) kpc

    # Timing-model parameters
    fit_param_names: Optional[list] = None
    fit_param_values: Optional[np.ndarray] = None
    fit_param_uncertainties: Optional[np.ndarray] = None
    Mmat_is_linear: Optional[np.ndarray] = None
    Mmat_linear: Optional[np.ndarray] = None        # derived; kept for old files
    Mmat_linear_labels: Optional[list] = None       # derived; kept for old files

    # NEW: par file, timing params to sample, ephemeris
    par_text: Optional[str] = None        # the par file itself (psr.write_par)
    sample_params: Optional[list] = None  # timing params for nonlinear sampling
    ephem: Optional[str] = None

    # NEW: all TOA flags, as names + one row per flag
    flag_names: Optional[np.ndarray] = None         # (n_flags,)
    flag_values: Optional[np.ndarray] = None        # (n_flags, n_toa) str

    # NEW: solar wind (DM for n_E = 1 cm^-3 at 1 AU) [pc]
    sw_geometry: Optional[np.ndarray] = None
    sw_geometry_method: Optional[str] = None        # 'pint' | 'approx' | ...

    # NEW: Adaptus / GTM basis for this pulsar
    adaptus_basis: Optional[np.ndarray] = None      # (n_toa, k)
    adaptus_mean: Optional[np.ndarray] = None       # (n_toa,)
    
    def __post_init__(self):
        s = lambda x: None if x is None else str(np.asarray(x).item() if np.ndim(x) == 0 else x)
        sl = lambda x: None if x is None else [str(v) for v in np.asarray(x).reshape(-1)]
        f = lambda x: None if x is None else np.asarray(x, dtype=np.float64)

        # strings come out of np.load as 0-d arrays; make them plain str
        for k in ('name', 'par_file', 'tim_file', 'timing_package', 'par_text',
                  'ephem', 'sw_geometry_method'):
            setattr(self, k, s(getattr(self, k)))
        for k in ('Mmat_labels', 'fit_param_names', 'sample_params'):
            setattr(self, k, sl(getattr(self, k)))
        for k in ('toas', 'residuals', 'toaerrs', 'freqs', 'pos', 'Mmat', 'pdist',
                  'fit_param_values', 'fit_param_uncertainties', 'sw_geometry',
                  'adaptus_basis', 'adaptus_mean'):
            setattr(self, k, f(getattr(self, k)))
        self.backend_flags = np.asarray(self.backend_flags).astype(str)
        if self.flag_names is not None:
            self.flag_names = np.asarray(self.flag_names).astype(str)
            self.flag_values = np.asarray(self.flag_values).astype(str)
        for k in ('raj', 'decj'):
            v = getattr(self, k)
            setattr(self, k, None if v is None else float(np.asarray(v)))

        # derived fields
        if self.pdist is None:            # PTA_Data always reads pdist; enterprise's placeholder
            self.pdist = np.array([1.0, 0.2])
            
        if self.raj is None or self.decj is None:
            x, y, z = self.pos
            self.raj, self.decj = float(np.arctan2(y, x) % (2 * np.pi)), float(np.arcsin(z))
        if self.Mmat_is_linear is None:
            self.Mmat_is_linear = [any(l.lower().startswith(p) for p in LINEAR_PARAMS)
                                   for l in self.Mmat_labels]
        self.Mmat_is_linear = np.asarray(self.Mmat_is_linear, dtype=bool)
        self.Mmat_linear = self.Mmat[:, self.Mmat_is_linear]
        self.Mmat_linear_labels = [l for l, b in zip(self.Mmat_labels, self.Mmat_is_linear) if b]
        if self.fit_param_names is None:
            self.fit_param_names = [l for l in self.Mmat_labels if l.lower() != 'offset']

        # checks
        n = self.toas.size
        for k in ('residuals', 'toaerrs', 'freqs', 'backend_flags', 'sw_geometry', 'adaptus_mean'):
            v = getattr(self, k)
            if v is not None and np.shape(v) != (n,):
                raise ValueError(f"{self.name}: {k} has shape {np.shape(v)}, expected ({n},)")
        if self.Mmat.shape != (n, len(self.Mmat_labels)):
            raise ValueError(f"{self.name}: Mmat {self.Mmat.shape} vs {n} TOAs / "
                             f"{len(self.Mmat_labels)} labels")
        if self.adaptus_basis is not None and self.adaptus_basis.shape[0] != n:
            raise ValueError(f"{self.name}: adaptus_basis has {self.adaptus_basis.shape[0]} rows")
        if self.flag_names is not None and self.flag_values.shape != (self.flag_names.size, n):
            raise ValueError(f"{self.name}: flag_values must be (n_flags, n_toa)")

    @property
    def fitpars(self):                    # enterprise's name; used by make_timing_model
        return list(self.Mmat_labels)

    @property
    def flags(self):                      # {flag: per-TOA str}
        if self.flag_names is None:
            return {}
        return {str(k): v for k, v in zip(self.flag_names, self.flag_values)}

    def write_par(self, directory):
        import os
        path = os.path.join(directory, f"{self.name}.par")
        os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as fh:
            fh.write(self.par_text)
        return path

    def save(self, path):
        """Write every non-None field; reload with PulsarDataLoader(**np.load(path))."""
        out = {}
        for k, v in asdict(self).items():
            if v is None or k in ('Mmat_linear', 'Mmat_linear_labels'):
                continue                  # derived on load
            if isinstance(v, (list, tuple)):
                v = np.asarray(v, dtype=str if k != 'Mmat_is_linear' else bool)
            out[k] = v
        np.savez_compressed(path, **out)

    @classmethod
    def from_enterprise(cls, psr, sw_geometry=None, **extra):
        """From an enterprise Pulsar (sort=False, drop_pintpsr=False).
        ``sw_geometry``: an array, or 'pint' to compute it from psr.model."""
        model = getattr(psr, 'model', None)
        names = [l for l in psr.fitpars if l.lower() != 'offset']
        kw = dict(name=psr.name, toas=psr.toas, residuals=psr.residuals,
                  toaerrs=psr.toaerrs, freqs=psr.freqs, pos=psr.pos,
                  backend_flags=psr.backend_flags, Mmat=psr.Mmat,
                  Mmat_labels=list(psr.fitpars), raj=psr._raj, decj=psr._decj,
                  pdist=np.asarray(getattr(psr, 'pdist', (1.0, 0.2)), float),
                  flag_names=np.array(list(psr.flags), dtype=str),
                  flag_values=np.array([np.asarray(v).astype(str) for v in psr.flags.values()]))
        if model is not None:
            kw.update(timing_package='pint', par_text=model.as_parfile(),
                      fit_param_names=names,
                      fit_param_values=[float(model[k].value) for k in names],
                      fit_param_uncertainties=[float(model[k].uncertainty_value or 0) for k in names])
        if isinstance(sw_geometry, str) and sw_geometry == 'pint':
            from ATLAS.signals.factorized.chromatic import sw_geometry_from_pint
            kw['sw_geometry'] = sw_geometry_from_pint(model, psr.pint_toas)
            kw['sw_geometry_method'] = 'pint'
        elif sw_geometry is not None:
            kw['sw_geometry'] = sw_geometry
        kw.update(extra)                  # e.g. sample_params=..., adaptus_basis=...
        return cls(**kw)



