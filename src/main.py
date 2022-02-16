"""
Main Code to reproduce the results in the paper
'Glottal Closure Instant Detection using Echo State Networks'.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3-Clause

import logging
from joblib import dump, load
from scipy.stats import uniform
from sklearn.utils.fixes import loguniform
import numpy as np

from pyrcn.echo_state_network import ESNRegressor
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.metrics import matthews_corrcoef
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, roc_curve, auc

from src.file_handling import get_file_list, train_test_split
from src.preprocessing import extract_features
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(context="paper", font="serif")


LOGGER = logging.getLogger(__name__)


def main(fit_ridge=False, fit_mlp=False, fit_esn=False, plot=False,
         serialize=False):
    """
    This is the main function to reproduce all visualizations and models for
    the paper "Glottal Closure Instant Detection using Echo State Networks".

    It is controlled via command line arguments:

    Params
    ------
    fit_ridge : bool, default=False
        Fit ridge regression models.
    fit_mlp : bool, default=False
        Whether to fit MLP models.
    fit_esn : bool, default=False
        Whether to fit ESN models.
    plot : bool, default=False
        Plot a reference and estimated GCI output.
    serialize:
        Store the fitted model in ``data/model.joblib``
    """

    LOGGER.info("Loading the training dataset...")
    audio_files = get_file_list("../data/SpLxDataLondonStudents2008/M/")
    LOGGER.info("... done!")

    LOGGER.info("Splitting dataset in training and test subsets...")
    training_files, test_files = train_test_split(audio_files)
    LOGGER.info("... done!")

    if fit_ridge:
        LOGGER.info("Fit and evaluate Ridge Regression models...")
        if plot:
            fig, axs = plt.subplots()
        for frame_length in [1, 3, 5, 7, 9, 11, 21, 31, 41, 51, 81]:
            feature_extraction_params = {
                "sr": 4000., "frame_length": frame_length}
            X_train, X_test, y_train, y_test = extract_features(
                training_files, test_files, target_widening=True,
                **feature_extraction_params)
            try:
                model = load(f"../results/ridge_"
                             f"{feature_extraction_params['frame_length']}"
                             f".joblib")
            except FileNotFoundError:
                model = RandomizedSearchCV(
                    estimator=Ridge(), param_distributions={
                        'alpha': loguniform(1e-5, 1e1)}, n_iter=20,
                    scoring="neg_mean_squared_error", n_jobs=-1,
                    random_state=42).fit(np.concatenate(X_train),
                                         np.concatenate(y_train))
                if serialize:
                    dump(model, f"../results/ridge_"
                         f"{feature_extraction_params['frame_length']}.joblib")

            y_pred = model.predict(np.concatenate(X_test))
            fpr, tpr, _ = roc_curve(np.concatenate(y_test), y_pred)
            roc_auc = auc(fpr, tpr)
            if plot:
                axs.plot(fpr, tpr, lw=2,
                         label=f'ROC curve (area = {roc_auc}, '
                               f'window = {frame_length})')
        if plot:
            axs.set(xlabel='False Positive Rate', ylabel="True Positive Rate")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()
        LOGGER.info("... done!")

    if fit_mlp:
        LOGGER.info("Fit and evaluate MLP Regression models...")
        if plot:
            fig, axs = plt.subplots()
        for frame_length in [1, 3, 5, 7, 9, 11, 21, 31, 41, 81]:
            feature_extraction_params = {
                "sr": 4000., "frame_length": frame_length}
            X_train, X_test, y_train, y_test = extract_features(
                training_files, test_files, target_widening=True,
                **feature_extraction_params)
            try:
                model = load(f"../results/mlp_"
                             f"{feature_extraction_params['frame_length']}"
                             f".joblib")
            except FileNotFoundError:
                model = RandomizedSearchCV(
                    estimator=MLPRegressor(hidden_layer_sizes=(12, 10),
                                           verbose=True, random_state=42),
                    param_distributions={}, scoring="neg_mean_squared_error",
                    n_jobs=-1, random_state=42).fit(np.concatenate(X_train),
                                                    np.concatenate(y_train))
                if serialize:
                    dump(model, f"../results/mlp_"
                         f"{feature_extraction_params['frame_length']}.joblib")

            y_pred = model.predict(np.concatenate(X_test))
            fpr, tpr, _ = roc_curve(np.concatenate(y_test), y_pred)
            roc_auc = auc(fpr, tpr)
            if plot:
                axs.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc}, '
                                               f'window = {frame_length})')
        if plot:
            axs.set(xlabel='False Positive Rate', ylabel="True Positive Rate")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()
        LOGGER.info("... done!")

    if fit_esn:
        LOGGER.info("Fit and evaluate ESN Regression models...")

        step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=10),
                            'spectral_radius': uniform(loc=0, scale=2)}

        step2_esn_params = {'leakage': uniform(1e-5, 1e0)}
        step3_esn_params = {'bias_scaling': uniform(loc=0, scale=3)}
        step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

        kwargs_step1 = {
            'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': 3,
            'scoring': make_scorer(matthews_corrcoef)}
        kwargs_step2 = {
            'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': 3,
            'scoring': make_scorer(matthews_corrcoef)}
        kwargs_step3 = {
            'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': 3,
            'scoring': make_scorer(matthews_corrcoef)}
        kwargs_step4 = {
            'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': 3,
            'scoring': make_scorer(matthews_corrcoef)}

        searches = [
            ('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', RandomizedSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

        if plot:
            fig, axs = plt.subplots()
        for frame_length in [1, 3, 5, 7, 9, 11, 21, 31, 41, 81]:
            feature_extraction_params = {
                "sr": 4000., "frame_length": frame_length}
            X_train, X_test, y_train, y_test = extract_features(
                training_files, test_files, target_widening=True,
                **feature_extraction_params)
            initially_fixed_params = {
                'hidden_layer_size': 50,
                'k_in': X_train[0].shape[1] if X_train[0].shape[1] < 5 else 5,
                'input_scaling': 0.4, 'input_activation': 'identity',
                'bias_scaling': 0.0, 'spectral_radius': 0.0, 'leakage': 1.0,
                'k_rec': 10, 'reservoir_activation': 'tanh',
                'bidirectional': False, 'alpha': 1e-3, 'random_state': 42}
            base_esn = ESNRegressor(**initially_fixed_params)
            try:
                model = load(f"../results/esn_"
                             f"{feature_extraction_params['frame_length']}"
                             f".joblib")
            except FileNotFoundError:
                model = SequentialSearchCV(
                    base_esn, searches=searches).fit(X_train, y_train)
                if serialize:
                    dump(model, f"../results/esn_"
                         f"{feature_extraction_params['frame_length']}.joblib")

            y_pred = model.predict(np.concatenate(X_test))
            fpr, tpr, _ = roc_curve(np.concatenate(y_test), y_pred)
            roc_auc = auc(fpr, tpr)
            if plot:
                axs.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc},'
                                               f'window = {frame_length})')
    if plot:
        axs.set(xlabel='False Positive Rate', ylabel="True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    LOGGER.info("... done!")

    LOGGER.info("Predicting GCIs on the test set...")
    y_pred = model.predict(X_test)
    LOGGER.info("... done!")
    if plot:
        fig, axs = plt.subplots(2, 1)
        sns.scatterplot(y=y_pred[0].ravel(), ax=axs[0])
        sns.scatterplot(y=y_test[0].ravel(), ax=axs[1])
        plt.xlabel("Sample index")
        plt.ylabel("Estimated GCI")
        plt.title("Test data")
        plt.tight_layout()

    if plot:
        plt.show()
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_ridge", action="store_true")
    parser.add_argument("--fit_mlp", action="store_true")
    parser.add_argument("--fit_esn", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--serialize", action="store_true")
    args = vars(parser.parse_args())
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ])
    LOGGER.setLevel(logging.DEBUG)
    main(**args)
    exit(0)
