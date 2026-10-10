# Magellanic Dynamics

Interactive simulations of the Milky Way and Magellanic Clouds.

A Streamlit app based on `LMC-internal-orbits.ipynb`, with galaxy orbits and
massless test particles. Full live N-body simulations are planned for future
development.

## Run

With Python 3.12, from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL Streamlit prints. Set parameters in the sidebar, click Run
orbits, and explore 3D/2D paths, epochs, separations and speeds. Download inertial
coordinates in CSV/NPZ and the configuration in JSON. The most recent result
survives plot changes. Changes to physical inputs require another run.

## Scientific scope

The notebook supplies MW/LMC initial conditions, MilkyWayPotential2022, a
2e11 Msun Hernquist LMC with a 1 kpc scale, and Gaussian massless tracers.
The optional SMC has explicitly illustrative Cartesian initial conditions;
it is disabled by default. None of its values is a literature preset.
The MW potential parameters are Gala's defaults and are not adjustable in v1.

Gala's compiled DirectNBody integrator handles runs without friction. With
friction, public Gala acceleration/density APIs provide the forces and SciPy's
adaptive DOP853 integrates them. Both paths return a Gala Orbit.
This avoids private Gala N-body internals. The backend's integrate(config)
returns arrays, names, the used config, and a Gala Orbit for further analysis.
All saved times are relative to present-day t=0. Negative times are past times.
Output sampling is independent of the adaptive solver's internal time steps.

The notebook's negative Coulomb logarithm and reversed friction return combine
into ordinary drag. Here ln Lambda is positive and the acceleration explicitly
opposes LMC velocity relative to the MW. Friction is disabled by default, has
no MW recoil, and uses a constant velocity dispersion. Backward integrations
with this dissipative prescription are model-dependent reconstructions.
Rigid translated potentials and center-force evaluations are approximations,
not self-consistent live halos. LMC parameters were marked as made up in the
notebook; replace them and the SMC placeholders for scientific use. Stellar
tracers are a random cloud, not an equilibrium LMC stellar population.

## Verification

Run `python test_dynamics.py` for public force comparisons, friction sign,
fixed-MW behavior, forward/backward round-trip, SMC/tracer shape checks, and
sampling-convergence checks. Gala 1.12.0 was used for development.

Gala reference: https://gala.adrian.pw/en/v1.12.0/
