# Self-supervised Traffic Advisor

- SSTA_2_views.py: 2-view
- SSTA_4_views.py: 4-view
- SSTA_8_views.py: 8-view

## Requirements
- Python 3.7.5
- CARLA
  - CARLA 0.9.11
- PyG
```bash
conda env create -f i2v_pyg_environment.yml
```

### Setup CARLA
```
Download CARLA 0.9.11 from https://github.com/carla-simulator/carla/releases
wget -c https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz
HOME_DIR=$HOME
mkdir -p ${HOME_DIR}/App/CARLA_0.9.11/
mv CARLA_0.9.11.tar.gz ${HOME_DIR}/App/CARLA_0.9.11/
cd ${HOME_DIR}/App/CARLA_0.9.11/
tar xvf CARLA_0.9.11.tar.gz
```
Add the following variables to `~/.bashrc` or `~/.zshrc`:
```
export CARLA_ROOT=${HOME_DIR}/App/CARLA_0.9.11
export CARLA_SERVER=${HOME_DIR}/App/CARLA/CarlaUE4.sh
# ${CARLA_ROOT} is the CARLA installation directory
# ${SCENARIO_RUNNER} is the ScenarioRunner installation directory
# <VERSION> is the correct string for the Python version being used
# In a build from source, the .egg files may be in: ${CARLA_ROOT}/PythonAPI/dist/ instead of ${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```
Collect CARLA Data
```
cd ${CARLA_ROOT}  # ${HOME_DIR}/App/CARLA_0.9.11
./CarlaUE4.sh
cd ${Curr_DIR}  # Current SSTA code directory
ln -s /media/data/jack/Apps/CARLA_0.9.11/PythonAPI/carla .
cd carla_tools
python manual_control.py --autopilot
python spawn_npc.py -n 100 -w 0 --sync
# or manually record videos
python manual_control.py
```
Manual control without rendering
The script PythonAPI/examples/no_rendering_mode.py provides an overview of the simulation. It creates a minimalistic aerial view with Pygame, that will follow the ego vehicle. This could be used along with manual_control.py to generate a route with barely no cost, record it, and then play it back and exploit it to gather data.
```
cd /opt/carla/PythonAPI/examples
python3 manual_control.py
cd /opt/carla/PythonAPI/examples
python3 no_rendering_mode.py --no-rendering
# Press "i" toggle actor id
```
Finally, convert the recorded images to the format required for training
```
python img2video.py
```
## How to Run
Train the VAE first, and save the vae checkpoint to `$VAE_CKPT`.
```
python vae.py
```

Take `SSTA_2_views.py` as an example, 
```bash
python SSTA_2_view.py --vae_ckpt_dir $VAE_CKPT # Training
python SSTA_2_view.py --vae_ckpt_dir $VAE_CKPT --mode eval # Evaluation
```

### Utils
#### img2video
```
python img2video.py (img2video())
```


# Reference
**[Self-Supervised Traffic Advisors: Distributed, Multi-view Traffic Prediction for Smart Cities](https://arxiv.org/abs/2204.06171)**
<br />
[Jiankai Sun](https://scholar.google.com/citations?user=726MCb8AAAAJ&hl=en),
[Shreyas Kousik](https://www.shreyaskousik.com/), 
[David Fridovich-Keil](https://dfridovi.github.io/), and
[Mac Schwager](http://web.stanford.edu/~schwager/)
<br />
**In IEEE 25th International Conference on Intelligent Transportation Systems (ITSC) 2022**
<br />
[[Paper]](https://arxiv.org/abs/2204.06171)[[Code]](https://github.com/Jiankai-Sun/SSTA_ITSC_2022)

```
@ARTICLE{sun2022selfsupervised,
     author={J. {Sun} and S. {Kousik} and D. {Fridovich-Keil} and M. {Schwager}},
     journal={IEEE 25th International Conference on Intelligent Transportation Systems (ITSC)},
     title={Self-Supervised Traffic Advisors: Distributed, Multi-view Traffic Prediction for Smart Cities},
     year={2022},
}
```
