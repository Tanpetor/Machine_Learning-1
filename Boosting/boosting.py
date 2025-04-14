from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import KBinsDiscretizer
from scipy import sparse

from typing import Optional


class MinEntropyDiscretizer:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges: List[List[float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        self.bin_edges = []
        for feature_idx in range(n_features):
            X_feature = X[:, feature_idx]
            y_feature = y

            sorted_indices = np.argsort(X_feature)
            X_sorted = X_feature[sorted_indices]
            y_sorted = y_feature[sorted_indices]

            edges = [X_sorted[0]]
            bin_size = len(X_sorted) // self.n_bins
            for i in range(1, self.n_bins):
                index = i * bin_size
                if index < len(X_sorted):
                    edges.append(X_sorted[index])
            edges.append(X_sorted[-1])

            edges = sorted(list(set(edges)))
            self.bin_edges.append(edges)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_binned = np.zeros_like(X, dtype=int)
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            edges = self.bin_edges[feature_idx]
            X_binned[:, feature_idx] = np.digitize(X[:, feature_idx], bins=edges[1:-1], right=False)
        return X_binned

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class PiecewiseEncodingDiscretizer:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges: List[List[float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        self.bin_edges = []
        for feature_idx in range(n_features):
            X_feature = X[:, feature_idx]
            y_feature = y

            sorted_indices = np.argsort(X_feature)
            X_sorted = X_feature[sorted_indices]
            y_sorted = y_feature[sorted_indices]

            edges = [X_sorted[0]]
            cumulative = 0
            total = np.sum(y_sorted)
            target_per_bin = total / self.n_bins
            current_sum = 0
            for x, label in zip(X_sorted, y_sorted):
                current_sum += label
                if current_sum >= target_per_bin:
                    edges.append(x)
                    current_sum = 0
                    if len(edges) == self.n_bins:
                        break

            edges = sorted(list(set(edges)))
            if len(edges) < self.n_bins + 1:
                edges += [X_sorted[-1]] * (self.n_bins + 1 - len(edges))
            self.bin_edges.append(edges)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_binned = np.zeros_like(X, dtype=int)
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            edges = self.bin_edges[feature_idx]
            X_binned[:, feature_idx] = np.digitize(X[:, feature_idx], bins=edges[1:-1], right=False)
        return X_binned

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_class=DecisionTreeRegressor,
            base_model_params: Optional[dict] = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            early_stopping_rounds: int | None = None,
            subsample: float | int = 1.0,
            bagging_temperature: float | int = 1.0,
            bootstrap_type: str | None = 'Bernoulli',
            goss: bool | None = False,
            goss_k: float | int = 0.2,
            rsm: float | int = 1.0,
            quantization_type: str | None = None,
            nbins: int = 255,
            dart: bool | None = False,
            dropout_rate: float = 0.05
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: y * (1 - self.sigmoid(y * z))

        self.early_stopping_rounds = early_stopping_rounds
        self.best_val_score = None
        self.best_iteration = 0
        self.stopping_steps = 0

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type

        self.goss = goss
        self.goss_k = goss_k

        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins

        if self.quantization_type in ['uniform', 'quantile']:
            self.kbins = KBinsDiscretizer(n_bins=self.nbins, encode='ordinal', strategy=self.quantization_type)
        elif self.quantization_type == 'MinEntropy':
            self.kbins = MinEntropyDiscretizer(n_bins=self.nbins)
        elif self.quantization_type == 'PiecewiseEncoding':
            self.kbins = PiecewiseEncodingDiscretizer(n_bins=self.nbins)
        else:
            self.kbins = None

        self.selected_features: Optional[np.ndarray] = None

        self.dart = dart
        self.dropout_rate = dropout_rate
        self.active_models = []

    @property
    def feature_importances_(self):
        importances = np.array([model.feature_importances_ for model in self.models])
        avg_importances = np.mean(importances, axis=0)
        normalized_importances = avg_importances / avg_importances.sum()
        if self.selected_features is not None:
            total_features = self._get_total_features()
            all_importances = np.zeros(total_features)
            all_importances[self.selected_features] = normalized_importances
            return all_importances
        else:
            return normalized_importances

    def _get_total_features(self):
        return self.n_features_

    def _apply_quantization(self, X, y=None):
        if sparse.issparse(X):
            X = X.toarray()
        if self.kbins is not None:
            if isinstance(self.kbins, (MinEntropyDiscretizer, PiecewiseEncodingDiscretizer)):
                if y is not None:
                    X = self.kbins.fit_transform(X, y)
                else:
                    X = self.kbins.transform(X)
            else:
                if y is not None:
                    X = self.kbins.fit_transform(X, y)
                else:
                    X = self.kbins.transform(X)
        return X

    def _handle_feature_subsampling(self, X):
        if self.selected_features is not None:
            return X[:, self.selected_features]

        if self.rsm < 1.0:
            n_features = X.shape[1]
            if isinstance(self.rsm, float):
                n_sub_features = max(1, min(int(self.rsm * n_features), n_features))
            else:
                n_sub_features = min(self.rsm, n_features)
            self.selected_features = np.random.choice(n_features, n_sub_features, replace=False)
            X = X[:, self.selected_features]
        else:
            self.selected_features = np.arange(X.shape[1])

        return X

    def partial_fit(self, X, y, current_predictions):
        if self.selected_features is not None:
            X = X[:, self.selected_features]
        X = self._apply_quantization(X, y)

        if self.dart:
            n_models = len(self.models)
            if n_models > 0:
                k = max(1, int(self.dropout_rate * n_models))
                dropout_indices = np.random.choice(n_models, size=k, replace=False)

                active_models = [model for idx, model in enumerate(self.models) if idx not in dropout_indices]
                k = len(dropout_indices)
                if k > 0:
                    scaling_factor = k / (k + 1)
                    for idx in dropout_indices:
                        scaling = k / (k + 1)
                        current_predictions -= self.learning_rate * self.gammas[idx] * self.models[idx].predict(
                            X) * scaling
                else:
                    active_models = self.models

                agg_predictions = np.zeros(X.shape[0])
                active_gammas = [self.gammas[idx] for idx in range(len(self.models)) if idx not in dropout_indices]
                for model, gamma in zip(active_models, active_gammas):
                    agg_predictions += self.learning_rate * gamma * model.predict(X)

                gradients = self.loss_derivative(y, agg_predictions)
            else:
                gradients = self.loss_derivative(y, current_predictions)

            model = self.base_model_class(**self.base_model_params)
            model.fit(X, gradients)

            predictions = model.predict(X)
            gamma = self.find_optimal_gamma(y, current_predictions, predictions)

            current_predictions += self.learning_rate * gamma * predictions

            self.models.append(model)
            self.gammas.append(gamma)
        else:

            if self.goss:
                gradients = np.abs(self.loss_derivative(y, current_predictions))
                threshold = np.quantile(gradients, 1 - self.goss_k)
                large_grad_indices = np.where(gradients >= threshold)[0]
                small_grad_indices = np.where(gradients < threshold)[0]
                if len(small_grad_indices) > 0:
                    sampled_small_grad_indices = np.random.choice(
                        small_grad_indices,
                        size=int(self.subsample * len(small_grad_indices)),
                        replace=False
                    )
                    selected_indices = np.concatenate((large_grad_indices, sampled_small_grad_indices))
                else:
                    selected_indices = large_grad_indices

                total_samples = len(selected_indices)
                scaling_factor = len(large_grad_indices) / total_samples if total_samples > 0 else 1.0

                X_sampled, y_sampled = X[selected_indices], y[selected_indices]
                current_predictions_sampled = current_predictions[selected_indices] * scaling_factor
            else:
                if self.bootstrap_type == 'Bernoulli':
                    sample_size = int(self.subsample * X.shape[0])
                    sample_size = max(1, sample_size)
                    if sample_size > X.shape[0]:
                        sample_size = X.shape[0]
                    indices = np.random.choice(
                        np.arange(X.shape[0]),
                        size=sample_size,
                        replace=False
                    )
                elif self.bootstrap_type == 'Bayesian':
                    weights = (-np.log(np.random.uniform(size=X.shape[0]))) ** self.bagging_temperature
                    indices = np.random.choice(
                        np.arange(X.shape[0]),
                        size=int(self.subsample * X.shape[0]),
                        replace=True,
                        p=weights / weights.sum()
                    )

                X_sampled, y_sampled = X[indices], y[indices]
                current_predictions_sampled = current_predictions[indices]

            gradient = self.loss_derivative(y_sampled, current_predictions_sampled)
            model = self.base_model_class(**self.base_model_params)
            model.fit(X_sampled, gradient)

            predictions = model.predict(X)
            gamma = self.find_optimal_gamma(y, current_predictions, predictions)

            current_predictions += self.learning_rate * gamma * predictions

            self.models.append(model)
            self.gammas.append(gamma)

        return current_predictions

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool
        """
        self.n_features_ = X_train.shape[1]

        X_train = self._handle_feature_subsampling(X_train)
        X_train = self._apply_quantization(X_train, y_train)

        if X_val is not None and y_val is not None:
            X_val = X_val[:, self.selected_features]
            X_val = self._apply_quantization(X_val, y_val)

        start_time = time.time()
        train_predictions = np.zeros(y_train.shape[0])
        val_predictions = np.zeros(y_val.shape[0]) if X_val is not None and y_val is not None else None

        for i in range(self.n_estimators):
            train_predictions = self.partial_fit(X_train, y_train, train_predictions)

            train_loss = self.loss_fn(y_train, train_predictions)
            train_auc = score(self, X_train, y_train)

            self.history["train_loss"].append(train_loss)
            self.history["train_roc_auc"].append(train_auc)

            if X_val is not None and y_val is not None:
                val_predictions = self.predict_proba(X_val)[:, 1]
                val_loss = self.loss_fn(y_val, val_predictions)
                val_auc = roc_auc_score(y_val == 1, val_predictions)

                self.history["val_loss"].append(val_loss)
                self.history["val_roc_auc"].append(val_auc)

                if self.early_stopping_rounds is not None:
                    if self.best_val_score is None or val_auc > self.best_val_score:
                        self.best_val_score = val_auc
                        self.best_iteration = i
                        self.stopping_steps = 0
                    else:
                        self.stopping_steps += 1

                    if self.stopping_steps >= self.early_stopping_rounds:
                        break
        end_time = time.time()
        print(f"Total training time: {end_time - start_time:.2f} seconds")
        if plot:
            self.plot_history(X_train, y_train)

    def predict_proba(self, X):
        if self.selected_features is None:
            raise ValueError("Model has not been fitted yet.")

        X = X[:, self.selected_features]
        X = self._apply_quantization(X)

        if not self.models:
            return self.sigmoid(np.zeros(X.shape[0])).reshape(-1, 1)

        agg_predictions = np.zeros(X.shape[0])
        for model, gamma in zip(self.models, self.gammas):
            agg_predictions += self.learning_rate * gamma * model.predict(X)

        prob_class_1 = self.sigmoid(agg_predictions)
        prob_class_0 = 1 - prob_class_1
        return np.vstack((prob_class_0, prob_class_1)).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)

    def plot_history(self, X=None, y=None):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label='Train Loss')
        if "val_loss" in self.history:
            plt.plot(self.history["val_loss"], label='Validation Loss')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_roc_auc"], label='Train ROC AUC', color='orange')
        if "val_roc_auc" in self.history:
            plt.plot(self.history["val_roc_auc"], label='Validation ROC AUC', color='green')
        plt.xlabel('Number of Estimators')
        plt.ylabel('ROC AUC')
        plt.title('ROC AUC History')
        plt.legend()

        plt.tight_layout()
        plt.show()
