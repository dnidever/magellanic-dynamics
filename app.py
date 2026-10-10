import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dynamics import DEFAULTS, integrate

st.set_page_config(page_title='Magellanic Dynamics',page_icon='🌌',layout='wide')
st.title('Magellanic Dynamics')
st.caption('Milky Way • Large Magellanic Cloud • Small Magellanic Cloud | Gala potentials')
with st.sidebar:
    st.header('Orbit setup')
    with st.form('setup'):
        c = DEFAULTS.copy()
        c['direction'] = -1 if st.selectbox('Integrate',['Backward','Forward']) == 'Backward' else 1
        c['duration'] = st.number_input('Duration (Gyr)',0.01,10.,2.,0.1)
        c['step'] = st.number_input('Output interval (Myr)',0.1,100.,2.,0.5)
        st.caption('Adaptive DOP853 internal steps; this interval controls saved output.')
        c['moving_mw'] = st.checkbox('Allow Milky Way reflex motion',True)
        c['smc'] = st.checkbox('Include SMC',False)
        c['friction'] = st.checkbox('LMC dynamical friction',False)
        with st.expander('Potentials and friction'):
            st.caption('MW: Gala MilkyWayPotential2022. Clouds: Hernquist spheres.')
            c['lmc_mass'] = st.number_input('LMC mass (10¹⁰ M☉)',0.01,100.,20.)*1e10
            c['lmc_scale'] = st.number_input('LMC scale radius (kpc)',0.1,50.,1.)
            c['smc_mass'] = st.number_input('SMC mass (10⁹ M☉)',0.01,100.,3.)*1e9
            c['smc_scale'] = st.number_input('SMC scale radius (kpc)',0.1,50.,2.)
            c['ln_lambda'] = st.number_input('Coulomb logarithm ln Λ',0.,10.,0.4,0.1)
            c['sigma'] = st.number_input('MW velocity dispersion (km/s)',10.,500.,150.)
        with st.expander('Present-day coordinates'):
            st.caption('Cartesian frame matching the notebook. MW starts at rest at the origin. SMC values are illustrative, not a measured preset.')
            for body in ['lmc','smc']:
                for kind,unit in [('pos','kpc'),('vel','km/s')]:
                    cols=st.columns(3)
                    c[f'{body}_{kind}']=[cols[i].number_input(f'{body.upper()} {axis} ({unit})' if kind=='pos' else f'{body.upper()} v{axis} ({unit})',value=float(DEFAULTS[f'{body}_{kind}'][i]),key=f'{body}_{kind}_{i}') for i,axis in enumerate('xyz')]
        with st.expander('LMC test stars and precision'):
            c['stars']=st.number_input('Massless tracers',0,100,10)
            c['seed']=st.number_input('Random seed',0,100000,42)
            c['star_radius']=st.number_input('Position scatter per axis (kpc)',0.,10.,0.5)
            c['star_sigma']=st.number_input('Velocity scatter per axis (km/s)',0.,100.,5.)
            c['rtol']=float(st.selectbox('Relative integration tolerance',['1e-8','1e-9','1e-10']))
        run=st.form_submit_button('Run orbits',type='primary',width='stretch')
if run:
    try:
        with st.spinner('Integrating orbits…'):
            st.session_state.result=integrate(c)
    except Exception as e:
        st.error(f'Integration failed: {e}')
if 'result' not in st.session_state:
    st.info('Choose the setup and click Run orbits. Defaults reproduce the notebook’s gravitational setup without friction.')
    st.stop()
r=st.session_state.result
if c != r['config']:
    st.caption('Plots show the last completed run. Submit Run orbits to apply edited settings.')
frame=st.radio('Plot coordinates',['Relative to Milky Way','Inertial','Relative to LMC'],horizontal=True)
origin={'Relative to Milky Way':0,'Relative to LMC':1}.get(frame)
q=r['pos'].copy()
if origin is not None: q-=q[:,origin:origin+1,:]
show_stars=st.checkbox('Show LMC tracers',True)
indices=list(range(r['massive']))+(list(range(r['massive'],len(r['names']))) if show_stars else [])
colors=['#8a94a6','#f8a541','#61b8ff']
a,b,d=st.tabs(['Orbits','Distances and speeds','Data and assumptions'])
with a:
    projection=st.selectbox('Projection',['3D','XY','XZ','YZ'])
    epoch=st.slider('Saved epoch',0,len(r['t'])-1,0)
    st.caption(f'Marker epoch: {r["t"][epoch]:.3f} Gyr; paths cover the full integration.')
    fig=go.Figure()
    for j in indices:
        color=colors[j] if j<r['massive'] else '#798b9e'
        name=r['names'][j]; p=q[:,j,:]; marker=q[epoch,j,:]
        kw=dict(name=name,line=dict(color=color,width=3 if j<r['massive'] else 1),opacity=1 if j<r['massive'] else 0.35,showlegend=j<r['massive'])
        if projection=='3D':
            fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines',**kw))
            fig.add_trace(go.Scatter3d(x=[marker[0]],y=[marker[1]],z=[marker[2]],mode='markers',marker=dict(color=color,size=5),showlegend=False,name=name))
        else:
            x,y=[{'X':0,'Y':1,'Z':2}[s] for s in projection]
            fig.add_trace(go.Scatter(x=p[:,x],y=p[:,y],mode='lines',**kw))
            fig.add_trace(go.Scatter(x=[marker[x]],y=[marker[y]],mode='markers',marker=dict(color=color,size=9),name=name,showlegend=False))
    fig.update_layout(height=650,scene=dict(xaxis_title='X (kpc)',yaxis_title='Y (kpc)',zaxis_title='Z (kpc)',aspectmode='data'),margin=dict(l=10,r=10,t=20,b=10))
    if projection!='3D': fig.update_layout(xaxis_title=projection[0]+' (kpc)',yaxis_title=projection[1]+' (kpc)',yaxis=dict(scaleanchor='x',scaleratio=1))
    st.plotly_chart(fig,width='stretch')
with b:
    sep=go.Figure(); speed=go.Figure()
    pairs=[(1,0)] + ([(2,0),(2,1)] if r['config']['smc'] else [])
    rows=[]
    for j,k in pairs:
        dist=np.linalg.norm(r['pos'][:,j]-r['pos'][:,k],axis=1)
        vel=np.linalg.norm(r['vel'][:,j]-r['vel'][:,k],axis=1)
        name=f'{r["names"][j]} / {r["names"][k]}'
        sep.add_trace(go.Scatter(x=r['t'],y=dist,name=name))
        speed.add_trace(go.Scatter(x=r['t'],y=vel,name=name))
        imin=np.argmin(dist)
        rows.append(dict(pair=name,minimum_sampled_separation_kpc=dist[imin],epoch_Gyr=r['t'][imin]))
    sep.update_layout(xaxis_title='Time (Gyr)',yaxis_title='Separation (kpc)')
    speed.update_layout(xaxis_title='Time (Gyr)',yaxis_title='Relative speed (km/s)')
    st.plotly_chart(sep,width='stretch');st.plotly_chart(speed,width='stretch')
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    st.caption('Minimum separations are sampled minima within the chosen interval, not refined pericenters.')
with d:
    rows=[]
    for j,name in enumerate(r['names']):
        f=pd.DataFrame(np.column_stack([r['t'],r['pos'][:,j],r['vel'][:,j]]),columns=['time_Gyr','x_kpc','y_kpc','z_kpc','vx_kms','vy_kms','vz_kms'])
        f.insert(1,'body',name);rows.append(f)
    df=pd.concat(rows,ignore_index=True)
    st.download_button('Download orbit CSV (inertial)',df.to_csv(index=False),'magellanic_orbits.csv','text/csv')
    st.download_button('Download run configuration',json.dumps(r['config'],indent=2),'orbit_config.json','application/json')
    buffer=io.BytesIO();np.savez_compressed(buffer,time_Gyr=r['t'],position_kpc=r['pos'],velocity_kms=r['vel'],names=np.array(r['names']),config_json=json.dumps(r['config']))
    st.download_button('Download arrays (NPZ)',buffer.getvalue(),'magellanic_orbits.npz')
    st.dataframe(df.head(50),hide_index=True)
    st.markdown('''**Model assumptions**

- Rigid translating potentials; no tidal mass loss, halo deformation, gas, or cosmological evolution.
- MW responds to the clouds when reflex motion is enabled. Massless tracers do not affect the galaxies.
- Each center feels the other potentials evaluated at that center; this is not a self-consistent extended-body simulation.
- Optional Chandrasekhar friction acts on the LMC using its velocity and position relative to the moving MW, a constant dispersion, and the MW density. No friction recoil is added to the MW. Friction therefore breaks momentum conservation.
- Drag opposes relative velocity in forward time. Backward integration naturally reverses dissipation and is sensitive to assumptions.
- LMC mass and radius come from the notebook, where they are described as made up. SMC coordinates, mass and radius are illustrative. Replace these before interpreting histories scientifically.
- The initial coordinate axes match the supplied notebook; no observational coordinate transformation is applied.''')
    st.json(r['config'])
