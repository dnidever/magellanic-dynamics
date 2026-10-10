import io
import json
import time
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dynamics import DEFAULTS, integrate

st.set_page_config(page_title='Magellanic Dynamics',page_icon='🌌',layout='wide')
@st.cache_data(max_entries=8, show_spinner=False)
def cached_integrate(config):
    started = time.perf_counter()
    result = integrate(config)
    result.pop('orbit')  # UI uses arrays; keep the cache small and serializable.
    result['integration_seconds'] = time.perf_counter() - started
    return result

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
            c['stars']=st.number_input('Massless tracers',0,100,0)
            st.caption('Start with zero for fast galaxy orbits. Tight stellar orbits require many internal steps. Friction uses a slower Python solver.')
            c['seed']=st.number_input('Random seed',0,100000,42)
            c['star_radius']=st.number_input('Position scatter per axis (kpc)',0.,10.,0.5)
            c['star_sigma']=st.number_input('Velocity scatter per axis (km/s)',0.,100.,5.)
            c['rtol']=float(st.selectbox('Relative integration tolerance',['1e-8','1e-9','1e-10']))
        run=st.form_submit_button('Run orbits',type='primary',width='stretch')
if run:
    try:
        with st.spinner('Integrating orbits…'):
            st.session_state.result=cached_integrate(c)
    except Exception as e:
        st.error(f'Integration failed: {e}')
if 'result' not in st.session_state:
    st.info('Choose the setup and click Run orbits. Defaults reproduce the notebook’s gravitational setup without friction.')
    st.stop()
r=st.session_state.result
st.caption(f"Integration: {r['integration_seconds']:.2f} s on the server. Identical setups reuse cached results.")
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
    projection=st.selectbox('Projection',['All three (2D)','3D without WebGL','Interactive 3D (WebGL)','XY','XZ','YZ'])
    epoch=st.slider('Saved epoch',0,len(r['t'])-1,0)
    st.caption(f'Marker epoch: {r["t"][epoch]:.3f} Gyr; paths cover the full integration.')
    if projection == '3D without WebGL':
        left,right=st.columns(2)
        elevation=left.slider('Elevation (degrees)',-90,90,25)
        azimuth=right.slider('Azimuth (degrees)',-180,180,-60)
        fig3=plt.figure(figsize=(10,7))
        ax=fig3.add_subplot(111,projection='3d')
        for j in indices:
            color=colors[j] if j<r['massive'] else '#798b9e'
            p=q[:,j,:]; marker=p[epoch]
            ax.plot(*p.T,color=color,alpha=1 if j<r['massive'] else .35,linewidth=2 if j<r['massive'] else .7,label=r['names'][j] if j<r['massive'] else None)
            ax.scatter(*marker,color=color,s=25)
        bounds=q[:,indices,:].reshape(-1,3)
        middle=(bounds.min(axis=0)+bounds.max(axis=0))/2
        half=max(np.ptp(bounds,axis=0).max()/2,1.)
        ax.set(xlim=(middle[0]-half,middle[0]+half),ylim=(middle[1]-half,middle[1]+half),zlim=(middle[2]-half,middle[2]+half),xlabel='X (kpc)',ylabel='Y (kpc)',zlabel='Z (kpc)')
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=elevation,azim=azimuth);ax.legend()
        st.pyplot(fig3);plt.close(fig3)
        st.caption('This 3D view works without WebGL. Use the angle sliders to rotate it.')
    else:
        panels=['XY','XZ','YZ'] if projection=='All three (2D)' else [projection]
        is3d=projection=='Interactive 3D (WebGL)'
        fig=go.Figure() if is3d else make_subplots(rows=1,cols=len(panels),subplot_titles=panels,horizontal_spacing=.07)
        for col,panel in enumerate(panels,1):
            for j in indices:
                color=colors[j] if j<r['massive'] else '#798b9e'
                name=r['names'][j]; p=q[:,j,:]; marker=p[epoch]
                kw=dict(name=name,legendgroup=name,line=dict(color=color,width=3 if j<r['massive'] else 1),opacity=1 if j<r['massive'] else .35,showlegend=j<r['massive'] and col==1)
                if is3d:
                    fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines',**kw))
                    fig.add_trace(go.Scatter3d(x=[marker[0]],y=[marker[1]],z=[marker[2]],mode='markers',marker=dict(color=color,size=5),showlegend=False,name=name))
                else:
                    x,y=[{'X':0,'Y':1,'Z':2}[s] for s in panel]
                    fig.add_trace(go.Scatter(x=p[:,x],y=p[:,y],mode='lines',**kw),row=1,col=col)
                    fig.add_trace(go.Scatter(x=[marker[x]],y=[marker[y]],mode='markers',marker=dict(color=color,size=9),name=name,showlegend=False),row=1,col=col)
            if not is3d:
                fig.update_xaxes(title_text=panel[0]+' (kpc)',row=1,col=col)
                fig.update_yaxes(title_text=panel[1]+' (kpc)',scaleanchor='x' if col==1 else f'x{col}',scaleratio=1,row=1,col=col)
        fig.update_layout(height=650 if is3d else 500,scene=dict(xaxis_title='X (kpc)',yaxis_title='Y (kpc)',zaxis_title='Z (kpc)',aspectmode='data'),margin=dict(l=20,r=20,t=40,b=20),uirevision=frame)
        st.plotly_chart(fig,width='stretch',key='orbit_plot')
        if is3d:
            st.info('If WebGL fails, choose "3D without WebGL". For Chrome, enable graphics acceleration in Settings → System and relaunch. Check https://get.webgl.org/ to test WebGL.')
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
