Important commands to run to run this project folder

// for environment
* python3.11 -m venv .venv
* source .venv/bin/activate

// install libraries 
* pip install ipykernel mne numpy pandas scipy scikit-learn matplotlib seaborn tqdm

do this too
pip install -r requirements.txt

pip freeze > requirements.lock.txt

to run files:
python -m scripts.fetch_openneuro_s3

python -m scripts.build_p300_features

python -m scripts.train_baselines

python debug.py


always do
export PYTHONPATH="$(pwd)"


