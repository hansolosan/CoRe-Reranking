#!/usr/bin/env python3
"""
Modular trainer classes for head weight optimization.

This module provides different loss functions for learning attention head weights:
- BCETrainer: Binary Cross-Entropy with L1 regularization (logistic regression)
- InfoNCETrainer: Contrastive loss (InfoNCE) for listwise ranking

All trainers implement the same interface for easy swapping.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class BaseTrainer(ABC):
    """Abstract base class for head weight trainers."""

    def __init__(self, lambda_l1=0.01, max_iter=1000, random_state=42):
        """
        Initialize trainer.

        Args:
            lambda_l1: L1 regularization strength
            max_iter: Maximum iterations for optimization
            random_state: Random seed for reproducibility
        """
        self.lambda_l1 = lambda_l1
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights_ = None
        self.intercept_ = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this trainer."""
        pass

    @abstractmethod
    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit the model to training data.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            docs_per_query_train: Optional array of docs per query for listwise losses

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            probabilities: (n_samples,) probability of positive class
        """
        pass

    def predict(self, X, threshold=0.5):
        """
        Predict class labels.

        Args:
            X: Features (n_samples, n_features)
            threshold: Classification threshold

        Returns:
            predictions: (n_samples,) binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_weights(self):
        """
        Get learned head weights.

        Returns:
            weights: (n_features,) array of weights
        """
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.weights_

    def evaluate(self, X_val, y_val):
        """
        Evaluate model on validation data.

        Args:
            X_val: Validation features
            y_val: Validation labels

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_val)
        y_prob = self.predict_proba(X_val)

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.0,
            'num_nonzero_weights': int(np.sum(np.abs(self.weights_) > 1e-6)),
            'sparsity': 1.0 - np.sum(np.abs(self.weights_) > 1e-6) / len(self.weights_)
        }

        return metrics


class BCETrainer(BaseTrainer):
    """
    Binary Cross-Entropy trainer with L1 regularization.

    Uses sklearn's LogisticRegression with L1 penalty (Lasso).
    The loss is normalized by the number of samples:
        Loss = (1/n) * sum(BCE_loss) + lambda * ||w||_1
    """

    def __init__(self, lambda_l1=0.01, max_iter=1000, random_state=42, class_weight='balanced'):
        """
        Initialize BCE trainer.

        Args:
            lambda_l1: L1 regularization strength (normalized by num samples)
            max_iter: Maximum iterations for SAGA solver
            random_state: Random seed
            class_weight: Class weighting strategy ('balanced' or None)
        """
        super().__init__(lambda_l1, max_iter, random_state)
        self.class_weight = class_weight
        self._model = None

    @property
    def name(self) -> str:
        return "bce"

    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit logistic regression model.

        sklearn's objective is: ||w||_1 + C * sum(log_loss)
        We want normalized BCE: (1/n) * sum(log_loss) + lambda * ||w||_1

        These are equivalent when: C = 1 / (n * lambda)
        """
        n_samples = len(y_train)
        C = 1.0 / (n_samples * self.lambda_l1) if self.lambda_l1 > 0 else 1e6

        self._model = LogisticRegression(
            penalty='l1',
            C=C,
            solver='saga',
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight
        )

        self._model.fit(X_train, y_train)

        # Store weights for interface compatibility
        self.weights_ = self._model.coef_[0].copy()
        self.intercept_ = self._model.intercept_[0]

        return self

    def predict_proba(self, X):
        """Predict probability of positive class."""
        if self._model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._model.predict_proba(X)[:, 1]


class InfoNCETrainer(BaseTrainer):
    """
    InfoNCE (Contrastive) loss trainer for listwise ranking.

    Optimizes:
        L = -log(exp(w·x_pos) / (exp(w·x_pos) + sum(exp(w·x_neg))))

    With L1 regularization:
        Loss = (1/n_queries) * sum(InfoNCE) + lambda * ||w||_1

    This directly optimizes for ranking by treating each query's documents
    as a contrastive learning problem.
    """

    def __init__(self, lambda_l1=0.01, max_iter=1000, random_state=42,
                 learning_rate=0.01, temperature=1.0):
        """
        Initialize InfoNCE trainer.

        Args:
            lambda_l1: L1 regularization strength
            max_iter: Maximum iterations for optimization
            random_state: Random seed
            learning_rate: Learning rate for gradient descent
            temperature: Temperature for softmax scaling
        """
        super().__init__(lambda_l1, max_iter, random_state)
        self.learning_rate = learning_rate
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "infonce"

    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit model using InfoNCE loss with proximal gradient descent.

        Args:
            X_train: Training features (n_docs, n_features)
            y_train: Training labels (n_docs,) - 1 for positive, 0 for negative
            docs_per_query_train: Array of docs per query (required for InfoNCE)
        """
        if docs_per_query_train is None:
            raise ValueError("InfoNCE trainer requires docs_per_query_train")

        n_features = X_train.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Initialize weights
        self.weights_ = rng.randn(n_features) * 0.01
        self.intercept_ = 0.0

        # Group documents by query
        query_groups = []
        doc_offset = 0
        for n_docs in docs_per_query_train:
            query_groups.append({
                'X': X_train[doc_offset:doc_offset + n_docs],
                'y': y_train[doc_offset:doc_offset + n_docs]
            })
            doc_offset += n_docs

        # Proximal gradient descent
        for iteration in range(self.max_iter):
            grad = np.zeros(n_features)

            for group in query_groups:
                X_q = group['X']
                y_q = group['y']

                # Compute scores
                scores = X_q @ self.weights_ / self.temperature

                # Softmax probabilities
                scores_exp = np.exp(scores - scores.max())  # Numerical stability
                probs = scores_exp / scores_exp.sum()

                # Gradient: sum over docs of (prob - target) * x
                # For InfoNCE, target is 1 for positive docs, 0 for negative
                targets = y_q / max(y_q.sum(), 1)  # Normalize targets
                grad += X_q.T @ (probs - targets) / self.temperature

            # Average gradient
            grad /= len(query_groups)

            # Gradient step
            self.weights_ -= self.learning_rate * grad

            # Proximal step (soft thresholding for L1)
            self.weights_ = np.sign(self.weights_) * np.maximum(
                np.abs(self.weights_) - self.learning_rate * self.lambda_l1, 0
            )

        return self

    def predict_proba(self, X):
        """Predict scores (higher = more likely positive)."""
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        scores = X @ self.weights_
        # Convert to probabilities using sigmoid
        return 1 / (1 + np.exp(-scores))


# Registry of available trainers
TRAINERS = {
    'bce': BCETrainer,
    'infonce': InfoNCETrainer,
}


def get_trainer(name, **kwargs):
    """
    Factory function to get a trainer by name.

    Args:
        name: Trainer name ('bce', 'infonce')
        **kwargs: Arguments to pass to trainer constructor

    Returns:
        trainer: Instance of the requested trainer
    """
    name = name.lower()
    if name not in TRAINERS:
        raise ValueError(f"Unknown trainer: {name}. Available: {list(TRAINERS.keys())}")
    return TRAINERS[name](**kwargs)


def list_trainers():
    """Return list of available trainer names."""
    return list(TRAINERS.keys())
