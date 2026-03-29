# EDEN: Early Dark Energy in Nucleosynthesis

A Python package for modeling and analyzing Early Dark Energy (EDE) using Big Bang Nucleosynthesis (BBN) constraints and cosmological analysis.

## Overview

EDEN provides a framework for exploring various Early Dark Energy models during the primordial universe. The package integrates with the PRyMordial BBN code and provides tools for parameter upper limit inference, visualization, and analysis of cosmological observables.

## Project Structure

```
eden/
├── eden_model.py              # Core EDE model classes and registry
├── model.py                   # Base model implementations
├── ns.py                      # Nested sampling analysis tools
├── hubble_analysis.py         # Hubble tension and cosmological analysis
├── t_of_T.py                  # Temperature evolution calculations
├── plot_ns.py                 # Visualization and plotting utilities
├── sbbn_samples.csv           # Standard BBN sample data
├── PRyM/                      # Big Bang Nucleosynthesis solver submodule
├── PRyMrates/                 # BBN reaction rate data
├── pyproject.toml             # Project configuration
└── uv.lock                    # Dependency lock file
``` 

## Key Components

### Core Models (`eden_model.py`)

The package provides four main EDE models:

1. **CCModel** - Cosmological Constant-like model
   - Static dark energy density (Λ)
   - Parameters: `Lambda_MeV4`

2. **LinearModel** - Linear scaling with scale factor
   - Evolves with power law in scale factor
   - Parameters: `rho0_MeV4`, `w` (equation of state)

3. **TempDependentModel** - Temperature-dependent EDE
   - Energy density traces temperature evolution
   - Parameters: `rho0_MeV4`, `alpha` (temperature dependence)

4. **PolytropicModel** - Polytropic equation of state
   - Smooth transition between radiation and matter domination
   - Parameters: `a_t` (transition scale), `rho_t_MeV4` (plateau density)

All models include BBN nuisance parameters:
- `tau_n`: Neutron lifetime
- `Omegabh2`: Baryon density
- `p_npdg`: N→p decay parameter
- `p_dpHe3g`: Deuteron photodisintegration parameter

### Analysis Tools

- **`hubble_analysis.py`** - Study Hubble tension and cosmological parameter constraints
- **`ns.py`** - Nested sampling for Bayesian parameter inference
- **`plot_ns.py`** - Comprehensive plotting and visualization functions
- **`t_of_T.py`** - Temperature-time relationships and EDE evolution

### Data

- **`sbbn_samples.csv`** - Pre-computed Standard BBN samples for comparison

## Installation

```bash
# Clone the repository
git clone https://github.com/croi900/eden.git
cd eden

# Install with uv (recommended)
uv sync
```

### Requirements

- Python ≥ 3.13
- Core dependencies: NumPy, SciPy, Pandas
- Scientific sampling: Cobaya, Dynesty, emcee
- Analysis: GetDist, corner
- Parallel processing: Dask, joblib, schwimmbad
- Visualization: Matplotlib, Seaborn

### Parameter Inference

Use the nested sampling tools in `ns.py` for Bayesian parameter estimation with Dynesty.

### Visualization

Create publication-quality plots using functions in `plot_ns.py`:
- Corner plots of posterior distributions
- Model comparison visualizations

### Citations

```bibtex
@article{Burns:2024prymordial,
    author = "Burns, Anne-Katherine and Tait, Tim M. P. and Valli, Mauro",
    title = "{PRyMordial: the first three minutes, within and beyond the standard model}",
    doi = "10.1140/epjc/s10052-024-12442-0",
    journal = "Eur. Phys. J. C",
    volume = "84",
    number = "1",
    pages = "86",
    year = "2024"
}

@article{Speagle:2020,
    author = {Speagle, Joshua S.},
    title = "{dynesty: a dynamic nested sampling package for estimating Bayesian posteriors and evidences}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {493},
    number = {3},
    pages = {3132-3158},
    year = {2020},
    month = {02},
    issn = {0035-8711},
    doi = {10.1093/mnras/staa278},
    url = {https://doi.org/10.1093/mnras/staa278}
}

@software{Koposov:2024,
    author       = {Sergey Koposov and others},
    title        = {joshspeagle/dynesty: v2.1.4},
    month        = jun,
    year         = 2024,
    publisher    = {Zenodo},
    version      = {v2.1.4},
    doi          = {10.5281/zenodo.12537467},
    url          = {https://doi.org}
}

@inproceedings{Skilling:2004,
    author = {Skilling, John},
    title = "{Nested Sampling}",
    booktitle = {Bayesian Inference and Maximum Entropy Methods in Science and Engineering: 24th International Workshop},
    series = {AIP Conference Proceedings},
    volume = {735},
    pages = {395-405},
    year = {2004},
    doi = {10.1063/1.1835238},
    url = {https://doi.org}
}

@article{Skilling:2006,
    author = {Skilling, John},
    title = "{Nested sampling for general Bayesian computation}",
    journal = {Bayesian Analysis},
    volume = {1},
    number = {4},
    pages = {833--859},
    year = {2006},
    doi = {10.1214/06-BA127},
    url = {https://doi.org/10.1214/06-BA127}
}

@article{Feroz:2009,
    author = {Feroz, F. and Hobson, M. P. and Bridges, M.},
    title = "{MULTINEST: an efficient and robust Bayesian inference tool for cosmology and particle physics}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {398},
    number = {4},
    pages = {1601-1614},
    year = {2009},
    month = {10},
    issn = {0035-8711},
    doi = {10.1111/j.1365-2966.2009.14548.x},
    url = {https://doi.org}
}
```



