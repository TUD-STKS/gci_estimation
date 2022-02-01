# Glottal Closure Instant Detection using Echo State Networks
## Metadata
- Author: [Peter Steiner](mailto:peter.steiner@tu-dresden.de)
- Conference: Studientexte zur Sprachkommunikation:
Elektronische Sprachsignalverarbeitung 2021, TUDpress, Dresden, Germany
- Weblink: [Paper](https://www.vocaltractlab.de/publications/steiner-2021-essv.pdf),
[Repository](https://github.com/TUD-STKS/gci_estimation)

## Summary and Contents
This repository provides the code for our experiments about the detection of Glottal 
Closure Instants (GCI), which are the points in time at which the vocal folds close 
during the production of voiced speech. So far, this code can be used to train the 
different considered models (Ridge Regression, Echo State Network, Multilayer 
Perceptron). Comparative plots and the noise analysis of the models follow soon.

Please note that, due to significant updates of the PyRCN library, the results can 
slightly differ from the reported values in the paper. However, the main conclusions 
are still the same.

## File list
- The following scripts are provided in this repository
    - `scripts/run.sh`: UNIX Bash script to reproduce the Figures in the paper.
    - `scripts/run.ps2`: Windows PowerShell script to reproduce the Figures in the paper.
- The following python code is provided in `src`
    - `src/file_handling.py`: Utility functions for storing and loading data and models.
    - `src/preprocessing.py`: Utility functions for preprocessing the dataset.
    - `src/main.py`: The main script to run.
- `requirements.txt`: Text file containing all required Python modules to be installed.
- `README.md`: The README displayed here.
- `LICENSE`: Textfile containing the license for this source code. You can find 
- `data/`: The optional directory `data` contains
    - `SpLxDataLondonStudents2008/`: Training and test audio files. Please ask 
    [Peter Steiner](mailto:peter.steiner@tu-dresden.de) to obtain the dataset.
- `results/`
    - (Pre)-trained modelss.
- `.gitignore`: Command file for Github to ignore files with specific extensions.

## Usage

To manually reproduce the results, you should create a new Python venv as well.
Therefore, you can run the script `create_venv.sh` on a UNIX bash or `create_venv.ps1`
that will automatically install all packages from PyPI. Afterwards, just type 
`source .virtualenv/bin/activate` in a UNIX bash or `.virtualenv/Scripts/activate.ps1`
in a PowerShell.

At first, we import required Python modules and load the dataset, which is already 
stored in `data`. 

```python
from file_handling import get_file_list, train_test_split


audio_files = get_file_list("../data/SpLxDataLondonStudents2008/M/")
training_files, test_files = train_test_split(audio_files)
```

Since the data is stored as a Pandas dataframe, we can theoretically multiple features. 
Here, we restrict the data features to the living area. With the function 
`select_features`, we obtain numpy arrays and the feature transformer that can also be
used for transforming the test data later. Next, we normalize them to zero mean and 
unitary variance.

```python
from preprocessing import extract_features


feature_extraction_params = {"sr": 4000., "frame_length": 8}
X_train, X_test, y_train, y_test = extract_features(
    training_files, test_files, target_widening=True,
    **feature_extraction_params)
```

We optimize a model using a random search.

```python
from scipy.stats import uniform
from sklearn.utils.fixes import loguniform

from pyrcn.echo_state_network import ESNRegressor
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


initially_fixed_params = {
    'hidden_layer_size': 50,
    'k_in': X_train[0].shape[1] if X_train[0].shape[1] < 5 else 5,
    'input_scaling': 0.4, 'input_activation': 'identity',
    'bias_scaling': 0.0, 'spectral_radius': 0.0, 'leakage': 1.0,
    'k_rec': 10, 'reservoir_activation': 'tanh',
    'bidirectional': False, 'alpha': 1e-3, 'random_state': 42}

step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=10),
                    'spectral_radius': uniform(loc=0, scale=2)}

step2_esn_params = {'leakage': uniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': uniform(loc=0, scale=3)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

kwargs_step1 = {
    'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(matthews_corrcoef)}
kwargs_step2 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(matthews_corrcoef)}
kwargs_step3 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(matthews_corrcoef)}
kwargs_step4 = {
    'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1,
    'scoring': make_scorer(matthews_corrcoef)}

searches = [
    ('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
    ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
    ('step3', RandomizedSearchCV, step3_esn_params, kwargs_step3),
    ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]
base_esn = ESNRegressor(**initially_fixed_params).fit(X_train, y_train)
model = SequentialSearchCV(
    base_esn, searches=searches).fit(X_train, y_train)
```

Finally, we estimate on the test data.

```python
y_pred = model.predict(X_test)
```

After you finished your experiments, please do not forget to deactivate the venv by 
typing `deactivate` in your command prompt.

The aforementioned steps are summarized in the script `main.py`. The easiest way to
reproduce the results is to either download and extract this Github repository in the
desired directory, open a Linux Shell and call `run.sh` or open a Windows PowerShell and
call `run.ps1`. 

In that way, again, a [Python venv](https://docs.python.org/3/library/venv.html) is 
created, where all required packages (specified by `requirements.txt`) are installed.
Afterwards, the script `main.py` is excecuted with all default arguments activated in
order to reproduce all results in the paper.

If you want to suppress any options, simply remove the particular option.

## Acknowledgements

This research was financed by Europäischer Sozialfonds (ESF) and the Free State of Saxony
(Application number: 100327771).

We thank Adrian Fourcin (University College London, UK) for giving us permission to use
the EGG dataset.


## License and Referencing
This program is licensed under the BSD 3-Clause License. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```latex
@InProceedings{src:Steiner-21c,
	title = {Glottal Closure Instant Detection using Echo State Networks},
	author = {Peter Steiner and Ian S. Howard and Peter Birkholz},
	year = {2021},
	pages = {161--168},
	keywords = {Oral},
	booktitle = {Studientexte zur Sprachkommunikation: Elektronische Sprachsignalverarbeitung 2021},
	editor = {Stefan Hillmann and Benjamin Weiss and Thilo Michael and Sebastian Möller},
	publisher = {TUDpress, Dresden},
	isbn = {978-3-95908-227-3}
}
```
