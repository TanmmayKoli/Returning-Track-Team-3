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

train on one specific dataset:
python -m scripts.build_p300_features --dataset ds003061 --reset

always do
export PYTHONPATH="$(pwd)"

commands for training and running different models:
All models:
python -m scripts.train_p300_models --models lda,logreg,svm,rf

Just LDA + SVM:
python -m scripts.train_p300_models --models lda,svm

Just LDA:
python -m scripts.train_p300_models --models lda

run plots file:
python -m scripts.plot_p300_results \
  --preds data/db/models/p300_erp_windows_v1/lda_test_predictions.parquet
