import numpy as np
import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
from dynamics import integrate,drag,VUNIT

r=integrate(dict(duration=.02,step=2.,stars=0,friction=False))
w=gd.PhaseSpacePosition(pos=r['pos'][0].T*u.kpc,vel=r['vel'][0].T*u.km/u.s)
p=[gp.MilkyWayPotential2022(),gp.HernquistPotential(m=2e11*u.Msun,c=1*u.kpc,units=galactic)]
o=gd.DirectNBody(w,p).integrate_orbit(dt=-2*u.Myr,n_steps=10)
np.testing.assert_allclose(r['pos'],o.xyz.to_value(u.kpc).transpose(1,2,0),atol=1e-5)
v=np.array([10.,20.,30.])*VUNIT
assert np.dot(drag(np.array([10.,20.,30.]),v,p[0],2e11,.4,150.),v)<0
f=integrate(dict(duration=.02,stars=0,moving_mw=False))
np.testing.assert_allclose(f['pos'][:,0],0,atol=1e-12)
b=integrate(dict(duration=.02,direction=1,stars=0,moving_mw=False,lmc_pos=f['pos'][-1,1].tolist(),lmc_vel=f['vel'][-1,1].tolist()))
np.testing.assert_allclose(b['pos'][-1,1],f['pos'][0,1],atol=1e-5)
s=integrate(dict(duration=.01,smc=True,stars=3,friction=True))
assert s['pos'].shape==(6,6,3)
fine=integrate(dict(duration=.02,step=1.,stars=0))
np.testing.assert_allclose(fine['pos'][::2],r['pos'],atol=1e-8)
# Notebook-duration integration exercises long baseline stability.
long=integrate(dict(duration=2.,stars=10))
assert np.isfinite(long['pos']).all()
print('PASS: Gala DirectNBody comparison, drag sign, fixed MW, reversibility, SMC/tracers, sampling, 2 Gyr run')
