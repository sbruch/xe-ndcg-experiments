# Build LightGBM and set up Virtualenv
```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build ; cd build
cmake ..
make -j64

cd ../python-package

sudo apt-get install virtualenv python3-venv
python3 -m venv /tmp/xendcg
source /tmp/xendcg/bin/activate
pip install scipy scikit-learn numpy wheel
python setup.py install
```

# Running Experiments
You may reproduce:

* `quality` experiments:

  ```bash
  python main.py --paths /path/to/data/*libsvm
  ```

* experiments with 20% of labels perturbed:

  ```bash
  python main.py --label_noise 0.2 --paths /path/to/data/*libsvm
  ```

* experiments where each query is augmented by 40% of its original list size:

  ```bash
  python main.py --negs_from_others 0.4 --paths /path/to/data/*libsvm
  ```

* experiments where 10 impressions are generated for each query
  with click probabilities conditioned on labels [.05, .3, .5, .7, .95]
  (i.e., documents with label 0 are clicked with 5% probability):

  ```bash
  python main.py --click_impressions 10 --click_probs .05 .3 .5 .7 .95 --paths /path/to/data/*libsvm
  ```
