# install python 3.6
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6

# Install deps for Python 3.6
conda create -n anpy36 python=3.6
source activate anpy36
pip install ipykernel
python -m ipykernel install --name py36-test --user
conda install pytorch torchvision -c pytorch
conda install bcolz -c conda-forge
pip install opencv-python graphviz sklearn_pandas sklearn tqdm isoweek pandas_summary

# Get code from latest git
git clone https://github.com/fastai/fastai.git

# Create new Jupyter config
cd ~

jupyter notebook --generate-config

key=$(python -c "from notebook.auth import passwd; print(passwd())")

cd ~
mkdir certs
cd certs
certdir=$(pwd)
openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.key -out mycert.pem

cd ~
sed -i "1 a\
c = get_config()\\
c.NotebookApp.certfile = u'$certdir/mycert.pem'\\
c.NotebookApp.ip = '*'\\
c.NotebookApp.open_browser = False\\
c.NotebookApp.password = u'$key'\\
c.NotebookApp.port = 8888" .jupyter/jupyter_notebook_config.py

# Run Jupyter
jupyter notebook --certfile=~/certs/mycert.pem --keyfile=~/certs/mycert.key