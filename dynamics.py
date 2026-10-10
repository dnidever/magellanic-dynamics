"""Rigid translating Gala potentials; coordinates kpc, Myr, Msun."""
import numpy as np
import astropy.units as u
from astropy.constants import G
import gala.potential as gp
import gala.dynamics as gd
from gala.units import galactic
from scipy.integrate import solve_ivp
from scipy.special import erf

DEFAULTS = dict(duration=2., step=2., direction=-1, moving_mw=True,
    smc=False, friction=False, ln_lambda=0.4, sigma=150.,
    lmc_mass=2e11, lmc_scale=1., smc_mass=3e9, smc_scale=2.,
    lmc_pos=[-0.6,-41.3,-27.1], lmc_vel=[-63.9,-213.8,206.6],
    smc_pos=[15.,-38.,-44.], smc_vel=[20.,-150.,150.],
    stars=10, seed=42, star_radius=0.5, star_sigma=5., rtol=1e-8)
VUNIT = (u.km/u.s).to(u.kpc/u.Myr)
GG = G.decompose(galactic).value

def drag(r, v, potential, mass, ln_lambda, sigma):
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        return np.zeros(3)
    rho = potential.density(r*u.kpc).to_value(u.Msun/u.kpc**3).item()
    X = speed/(np.sqrt(2)*sigma*VUNIT)
    # Series avoids cancellation near zero speed.
    bracket = 4/np.sqrt(np.pi)*(X**3/3-X**5/5) if X < 1e-3 else erf(X)-2*X/np.sqrt(np.pi)*np.exp(-X*X)
    return -4*np.pi*GG**2*mass*rho*ln_lambda*bracket*v/speed**3

def integrate(config):
    c = {**DEFAULTS, **config}
    if c['duration'] <= 0 or c['step'] <= 0 or c['direction'] not in (-1,1):
        raise ValueError('Duration and sampling step must be positive; direction must be ±1.')
    mw = gp.MilkyWayPotential(version='v2')
    potentials = [mw, gp.HernquistPotential(m=c['lmc_mass']*u.Msun, c=c['lmc_scale']*u.kpc, units=galactic)]
    names = ['Milky Way', 'LMC']
    pos = [np.zeros(3), np.asarray(c['lmc_pos'])]
    vel = [np.zeros(3), np.asarray(c['lmc_vel'])]
    if c['smc']:
        potentials.append(gp.HernquistPotential(m=c['smc_mass']*u.Msun,c=c['smc_scale']*u.kpc,units=galactic))
        names.append('SMC'); pos.append(np.asarray(c['smc_pos'])); vel.append(np.asarray(c['smc_vel']))
    massive = len(names)
    rng = np.random.default_rng(c['seed'])
    offsets = rng.normal(0,c['star_radius'],(3,c['stars'])).T
    velocities = rng.normal(0,c['star_sigma'],(3,c['stars'])).T
    for i in range(c['stars']):
        pos.append(np.asarray(c['lmc_pos'])+offsets[i])
        vel.append(np.asarray(c['lmc_vel'])+velocities[i])
        names.append(f'LMC tracer {i+1}')
    n = len(names)
    initial = np.stack([np.array(pos),np.array(vel)*VUNIT])
    def rhs(t,y):
        q,v = y.reshape(2,n,3)
        a = np.zeros_like(q)
        for j,p in enumerate(potentials):
            targets = np.arange(n) != j
            relative = q[targets]-q[j]
            if np.any(np.linalg.norm(relative,axis=1) < 1e-8):
                raise ValueError('A body reached a potential cusp; change the initial conditions or model.')
            a[targets] += p.acceleration(relative.T*u.kpc).to_value(u.kpc/u.Myr**2).T
        if c['friction']:
            a[1] += drag(q[1]-q[0],v[1]-v[0],mw,c['lmc_mass'],c['ln_lambda'],c['sigma'])
        if not c['moving_mw']:
            a[0] = 0
        return np.stack([v,a]).ravel()
    end = c['direction']*c['duration']*1000
    samples = int(np.ceil(abs(end)/c['step']))+1
    if samples*n > 1_000_000:
        raise ValueError('Too many saved points. Increase the sampling interval or reduce tracers/duration.')
    times = np.linspace(0,end,samples)
    if not c['friction']:
        # Gala's compiled N-body solver is much faster for tracer-rich runs.
        start = gd.PhaseSpacePosition(pos=initial[0].T*u.kpc,vel=initial[1].T*u.kpc/u.Myr)
        if c['moving_mw']:
            nb = gd.DirectNBody(start,potentials+[None]*c['stars'])
            raw = nb.integrate_orbit(t=times*u.Myr,Integrator_kwargs={'rtol':c['rtol'],'atol':c['rtol']*1e-2})
            state = np.stack([raw.xyz.to_value(u.kpc).transpose(1,2,0),raw.v_xyz.to_value(u.kpc/u.Myr).transpose(1,2,0)])
        else:
            nb = gd.DirectNBody(start[1:],potentials[1:]+[None]*c['stars'],external_potential=mw)
            raw = nb.integrate_orbit(t=times*u.Myr,Integrator_kwargs={'rtol':c['rtol'],'atol':c['rtol']*1e-2})
            state = np.zeros((2,len(times),n,3))
            state[0,:,1:] = raw.xyz.to_value(u.kpc).transpose(1,2,0)
            state[1,:,1:] = raw.v_xyz.to_value(u.kpc/u.Myr).transpose(1,2,0)
    else:
        sol = solve_ivp(rhs,(0,end),initial.ravel(),method='DOP853',t_eval=times,rtol=c['rtol'],atol=c['rtol']*1e-2)
        if not sol.success:
            raise RuntimeError(sol.message)
        state = sol.y.reshape(2,n,3,-1).transpose(0,3,1,2)
    if not np.isfinite(state).all():
        raise RuntimeError('Integration produced nonfinite coordinates.')
    orbit = gd.Orbit(pos=state[0].transpose(2,0,1)*u.kpc,
                     vel=state[1].transpose(2,0,1)*u.kpc/u.Myr,t=times*u.Myr)
    return dict(t=times/1000,pos=state[0],vel=state[1]/VUNIT,names=names,massive=massive,config=c,orbit=orbit)
