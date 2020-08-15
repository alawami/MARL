conda install -c conda-forge -y tensorboardX 
apt update
apt install -y build-essential
apt install -y libopenmpi-dev
pip install mpi4py
pip install progressbar

pip install stable_baselines

pip install gym==0.10.5
#pip install baselines==0.1.4

# In baselines:
# python setup.py install

apt install libgl1-mesa-glx
