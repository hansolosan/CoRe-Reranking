#!/usr/bin/env python3
"""
Modular trainer classes for head weight optimization.

This module provides different loss functions for learning attention head weights:
- BCETrainer: Binary Cross-Entropy with L1 regularization (logistic regression)
- InfoNCETrainer: Contrastive loss (InfoNCE) for listwise ranking

All trainers implement the same interface for easy swapping.
"""

from abc import ABC, abstractmethod
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class BaseTrainer(ABC):
    """Abstract base class for head weight trainers with temperature-scaled softmax."""

    def __init__(self, lambda_l1=0.01, temperature=1.0, max_iter=1000, random_state=42, verbose=False):
        """
        Initialize trainer.

        Args:
            lambda_l1: L1 regularization strength
            temperature: Temperature for softmax over features. Lower values (e.g., 0.001)
                        make the feature distribution more peaked (few heads dominate).
                        Higher values make it more uniform. Default: 1.0 (standard softmax)
            max_iter: Maximum iterations for optimization
            random_state: Random seed for reproducibility
            verbose: Whether to print progress during training
        """
        self.lambda_l1 = lambda_l1
        self.temperature = temperature
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.weights_ = None
        self.intercept_ = None

    def _print_progress(self, iteration, max_iter, metrics=None, freq=None):
        """
        Print training progress.

        Args:
            iteration: Current iteration (0-indexed)
            max_iter: Maximum iterations
            metrics: Optional dict of metric_name -> value to display
            freq: Print frequency (default: 10% of max_iter, min 1, max 100)
        """
        if not self.verbose:
            return

        if freq is None:
            freq = max(1, min(100, max_iter // 10))

        if iteration % freq == 0 or iteration == max_iter - 1:
            pct = 100 * (iteration + 1) / max_iter
            msg = f"\r  [{self.name}] {iteration + 1}/{max_iter} ({pct:.0f}%)"
            if metrics:
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                msg += f" - {metrics_str}"
            print(msg, end="", flush=True)

            # Newline at the end
            if iteration == max_iter - 1:
                print()

    def _apply_temperature_softmax(self, X):
        """
        Apply temperature-scaled softmax to features.

        Like CoRe, applies softmax(features / temp) to normalize feature importance.

        Args:
            X: Features array of shape (n_samples, n_features) or (n_features,)

        Returns:
            Softmax-normalized features of the same shape as input
        """
        X_scaled = X / self.temperature

        if X.ndim > 1:
            # Batch processing: (n_samples, n_features)
            X_max = X_scaled.max(axis=1, keepdims=True)
            X_exp = np.exp(X_scaled - X_max)
            return X_exp / X_exp.sum(axis=1, keepdims=True)
        else:
            # Single sample: (n_features,)
            X_max = X_scaled.max()
            X_exp = np.exp(X_scaled - X_max)
            return X_exp / X_exp.sum()

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
    Binary Cross-Entropy trainer with L1 regularization and temperature-scaled softmax.

    Applies temperature-scaled softmax to features before training:
        features_normalized = softmax(features / T)
    Then uses sklearn's LogisticRegression with L1 penalty (Lasso).
    The loss is normalized by the number of samples:
        Loss = (1/n) * sum(BCE_loss) + lambda * ||w||_1
    """

    def __init__(self, lambda_l1=0.01, temperature=1.0, max_iter=1000, random_state=42, class_weight='balanced', verbose=False):
        """
        Initialize BCE trainer.

        Args:
            lambda_l1: L1 regularization strength (normalized by num samples)
            temperature: Temperature for softmax over features. Default: 1.0 (standard softmax)
            max_iter: Maximum iterations for SAGA solver
            random_state: Random seed
            class_weight: Class weighting strategy ('balanced' or None)
            verbose: Whether to print progress during training
        """
        super().__init__(lambda_l1, temperature, max_iter, random_state, verbose)
        self.class_weight = class_weight
        self._model = None

    @property
    def name(self) -> str:
        return "bce"

    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit logistic regression model with temperature-scaled softmax features.

        sklearn's objective is: ||w||_1 + C * sum(log_loss)
        We want normalized BCE: (1/n) * sum(log_loss) + lambda * ||w||_1

        These are equivalent when: C = 1 / (n * lambda)
        """
        if self.verbose:
            print(f"  [{self.name}] Fitting sklearn LogisticRegression (max_iter={self.max_iter})...", end=" ", flush=True)

        # Apply temperature-scaled softmax to features
        X_train_softmax = self._apply_temperature_softmax(X_train)

        n_samples = len(y_train)
        C = 1.0 / (n_samples * self.lambda_l1) if self.lambda_l1 > 0 else 1e6

        self._model = LogisticRegression(
            penalty='l1',
            C=C,
            solver='saga',
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            verbose=1 if self.verbose else 0
        )

        self._model.fit(X_train_softmax, y_train)

        # Store weights for interface compatibility
        self.weights_ = self._model.coef_[0].copy()
        self.intercept_ = self._model.intercept_[0]

        if self.verbose:
            n_nonzero = int(np.sum(np.abs(self.weights_) > 1e-6))
            print(f"done. Non-zero weights: {n_nonzero}")

        return self

    def predict_proba(self, X):
        """Predict probability of positive class with temperature-scaled softmax features."""
        if self._model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X_softmax = self._apply_temperature_softmax(X)
        return self._model.predict_proba(X_softmax)[:, 1]


class BCEWithTemperatureTrainer(BaseTrainer):
    """
    Binary Cross-Entropy trainer with temperature-scaled softmax features.

    Similar to CoRe's approach, this applies temperature scaling then softmax to the
    features (attention scores from each head) before linear combination:
        features_normalized = softmax(features / T)
        logit = w^T features_normalized
        p = sigmoid(logit)
        Loss = (1/n) * sum(-y*log(p) - (1-y)*log(1-p)) + lambda * ||w||_1

    This normalizes feature importance (like CoRe normalizes document scores) so that
    the most important heads are emphasized. Lower temperature makes the distribution
    more peaked (few heads dominate), higher temperature makes it more uniform.
    """

    def __init__(self, lambda_l1=0.01, temperature=1.0, max_iter=1000,
                 random_state=42, learning_rate=0.01, class_weight='balanced', verbose=False):
        """
        Initialize BCE trainer with temperature-scaled softmax features.

        Args:
            lambda_l1: L1 regularization strength
            temperature: Temperature for softmax over features. Lower values (e.g., 0.001)
                        make the feature distribution more peaked (few heads dominate).
                        Higher values (e.g., 10.0) make it more uniform.
                        Default: 1.0 (standard softmax)
            max_iter: Maximum iterations for gradient descent
            random_state: Random seed
            learning_rate: Learning rate for gradient descent
            class_weight: Class weighting strategy ('balanced' or None)
            verbose: Whether to print progress during training
        """
        super().__init__(lambda_l1, temperature, max_iter, random_state, verbose)
        self.learning_rate = learning_rate
        self.class_weight = class_weight

    @property
    def name(self) -> str:
        return "bce_temp"

    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit model using BCE loss with temperature-scaled softmax features.

        Like CoRe, applies softmax(features / temp) before combining with weights.
        Uses proximal gradient descent with soft thresholding for L1 regularization.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            docs_per_query_train: Not used (for interface compatibility)

        Returns:
            self
        """
        n_samples, n_features = X_train.shape
        rng = np.random.RandomState(self.random_state)

        # Initialize weights
        self.weights_ = rng.randn(n_features) * 0.01
        self.intercept_ = 0.0

        # Compute class weights
        if self.class_weight == 'balanced':
            n_pos = np.sum(y_train)
            n_neg = n_samples - n_pos
            if n_pos > 0 and n_neg > 0:
                w_pos = n_samples / (2 * n_pos)
                w_neg = n_samples / (2 * n_neg)
                sample_weights = np.where(y_train == 1, w_pos, w_neg)
            else:
                sample_weights = np.ones(n_samples)
        else:
            sample_weights = np.ones(n_samples)

        # Precompute softmax features (constant across iterations)
        X_softmax = self._apply_temperature_softmax(X_train)

        # Gradient descent with proximal step for L1
        for iteration in range(self.max_iter):
            # Compute logits with softmax-normalized features
            logits = X_softmax @ self.weights_ + self.intercept_
            probs = 1 / (1 + np.exp(-logits))  # sigmoid

            # Compute loss for progress
            eps = 1e-15
            loss = -np.mean(sample_weights * (y_train * np.log(probs + eps) + (1 - y_train) * np.log(1 - probs + eps)))

            self._print_progress(iteration, self.max_iter, {'loss': loss})

            # Compute gradient with respect to weights
            # Chain rule: d/dw = (p - y) * softmax(X/T)
            residuals = (probs - y_train) * sample_weights
            grad_weights = X_softmax.T @ residuals / n_samples
            grad_intercept = residuals.mean()

            # Gradient step
            self.weights_ -= self.learning_rate * grad_weights
            self.intercept_ -= self.learning_rate * grad_intercept

            # Proximal step (soft thresholding for L1)
            self.weights_ = np.sign(self.weights_) * np.maximum(
                np.abs(self.weights_) - self.learning_rate * self.lambda_l1, 0
            )

        return self

    def predict_proba(self, X):
        """
        Predict probability of positive class with temperature-scaled softmax features.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            probabilities: (n_samples,) probability of positive class
        """
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Apply same temperature-scaled softmax as in training
        X_softmax = self._apply_temperature_softmax(X)

        # Compute logits and probabilities
        logits = X_softmax @ self.weights_ + self.intercept_
        return 1 / (1 + np.exp(-logits))


class HingeLossTrainer(BaseTrainer):
    """
    Hinge loss trainer for pairwise ranking with Elastic Net regularization and temperature-scaled softmax.

    Applies temperature-scaled softmax to features before computing scores:
        features_normalized = softmax(features / T)

    Optimizes:
        L = (1/n_pairs) * sum(max(0, margin - (s_pos - s_neg))) + lambda_l1 * ||w||_1 + lambda_l2 * ||w||_2^2

    Combines:
    - Hinge loss: Margin-based separation between positive and negative docs
    - L1 regularization: Sparsity
    - L2 regularization: Stability and handling correlated features
    """

    def __init__(self, lambda_l1=0.01, lambda_l2=0.01, margin=1.0, temperature=1.0,
                 max_iter=1000, random_state=42, learning_rate=0.01, verbose=False):
        """
        Initialize Hinge loss trainer.

        Args:
            lambda_l1: L1 regularization strength (sparsity)
            lambda_l2: L2 regularization strength (stability)
            margin: Margin for hinge loss (positive should score at least margin higher than negative)
            temperature: Temperature for softmax over features. Default: 1.0 (standard softmax)
            max_iter: Maximum iterations for optimization
            random_state: Random seed
            learning_rate: Learning rate for gradient descent
            verbose: Whether to print progress during training
        """
        super().__init__(lambda_l1, temperature, max_iter, random_state, verbose)
        self.lambda_l2 = lambda_l2
        self.margin = margin
        self.learning_rate = learning_rate

    @property
    def name(self) -> str:
        return "hinge"

    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit model using pairwise hinge loss with temperature-scaled softmax features.

        Args:
            X_train: Training features (n_docs, n_features)
            y_train: Training labels (n_docs,) - 1 for positive, 0 for negative
            docs_per_query_train: Array of docs per query (required for pairwise loss)
        """
        if docs_per_query_train is None:
            raise ValueError("HingeLoss trainer requires docs_per_query_train")

        n_features = X_train.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Initialize weights
        self.weights_ = rng.randn(n_features) * 0.01
        self.intercept_ = 0.0

        # Apply temperature-scaled softmax to all features
        X_train_softmax = self._apply_temperature_softmax(X_train)

        # Create pairwise training examples
        pairs = []
        doc_offset = 0
        for n_docs in docs_per_query_train:
            X_q = X_train_softmax[doc_offset:doc_offset + n_docs]
            y_q = y_train[doc_offset:doc_offset + n_docs]

            pos_indices = np.where(y_q == 1)[0]
            neg_indices = np.where(y_q == 0)[0]

            # Create all positive-negative pairs for this query
            for pos_idx in pos_indices:
                for neg_idx in neg_indices:
                    pairs.append({
                        'x_pos': X_q[pos_idx],
                        'x_neg': X_q[neg_idx]
                    })

            doc_offset += n_docs

        if len(pairs) == 0:
            raise ValueError("No positive-negative pairs found in training data")

        # Gradient descent with Elastic Net
        for iteration in range(self.max_iter):
            grad = np.zeros(n_features)
            loss = 0.0

            for pair in pairs:
                x_pos = pair['x_pos']
                x_neg = pair['x_neg']

                # Scores
                s_pos = x_pos @ self.weights_
                s_neg = x_neg @ self.weights_

                # Hinge loss: max(0, margin - (s_pos - s_neg))
                violation = self.margin - (s_pos - s_neg)

                if violation > 0:
                    # Gradient: -( x_pos - x_neg )
                    grad += -(x_pos - x_neg)
                    loss += violation

            # Average over pairs
            grad /= len(pairs)
            loss /= len(pairs)

            self._print_progress(iteration, self.max_iter, {'loss': loss})

            # Add L2 gradient
            grad += 2 * self.lambda_l2 * self.weights_

            # Gradient step
            self.weights_ -= self.learning_rate * grad

            # Proximal step for L1 (soft thresholding)
            self.weights_ = np.sign(self.weights_) * np.maximum(
                np.abs(self.weights_) - self.learning_rate * self.lambda_l1, 0
            )

        return self

    def predict_proba(self, X):
        """Predict scores with temperature-scaled softmax features."""
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X_softmax = self._apply_temperature_softmax(X)
        scores = X_softmax @ self.weights_
        # Convert to probabilities using sigmoid
        return 1 / (1 + np.exp(-scores))


class ApproxNDCGTrainer(BaseTrainer):
    """
    ApproxNDCG trainer - differentiable approximation of NDCG.

    Optimizes:
        L = -NDCG_approx + lambda * ||w||_1

    Uses softmax to create differentiable approximation of ranking.
    This directly optimizes the evaluation metric (NDCG).
    """

    def __init__(self, lambda_l1=0.01, max_iter=1000, random_state=42,
                 learning_rate=0.01, temperature=1.0, k=10, verbose=False):
        """
        Initialize ApproxNDCG trainer.

        Args:
            lambda_l1: L1 regularization strength
            max_iter: Maximum iterations
            random_state: Random seed
            learning_rate: Learning rate
            temperature: Temperature for softmax (lower = sharper approximation)
            k: Compute NDCG@k
            verbose: Whether to print progress during training
        """
        super().__init__(lambda_l1, temperature, max_iter, random_state, verbose)
        self.learning_rate = learning_rate
        self.k = k

    @property
    def name(self) -> str:
        return "approx_ndcg"

    def _ndcg_at_k(self, scores, labels, k):
        """
        Compute differentiable approximation of NDCG@k.

        Uses softmax weights to approximate sorted positions.
        """
        # Sort by scores (true ranking)
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices][:k]

        # DCG
        positions = np.arange(1, min(k, len(sorted_labels)) + 1)
        discounts = np.log2(positions + 1)
        dcg = np.sum(sorted_labels / discounts)

        # IDCG (ideal ranking)
        ideal_labels = np.sort(labels)[::-1][:k]
        idcg = np.sum(ideal_labels / discounts[:len(ideal_labels)])

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit model by maximizing approx NDCG.

        Args:
            X_train: Training features (n_docs, n_features)
            y_train: Training labels (n_docs,)
            docs_per_query_train: Array of docs per query (required)
        """
        if docs_per_query_train is None:
            raise ValueError("ApproxNDCG trainer requires docs_per_query_train")

        n_features = X_train.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Initialize weights
        self.weights_ = rng.randn(n_features) * 0.01
        self.intercept_ = 0.0

        # Apply temperature-scaled softmax to features
        X_train_softmax = self._apply_temperature_softmax(X_train)

        # Group documents by query
        query_groups = []
        doc_offset = 0
        for n_docs in docs_per_query_train:
            query_groups.append({
                'X': X_train_softmax[doc_offset:doc_offset + n_docs],
                'y': y_train[doc_offset:doc_offset + n_docs]
            })
            doc_offset += n_docs

        # Gradient descent
        best_weights = self.weights_.copy()
        best_ndcg = -np.inf

        for iteration in range(self.max_iter):
            grad = np.zeros(n_features)
            total_ndcg = 0.0

            for group in query_groups:
                X_q = group['X']
                y_q = group['y']

                # Compute scores
                scores = X_q @ self.weights_

                # Compute NDCG for this query
                ndcg = self._ndcg_at_k(scores, y_q, self.k)
                total_ndcg += ndcg

                # Approximate gradient using finite differences
                epsilon = 1e-5
                for i in range(n_features):
                    w_plus = self.weights_.copy()
                    w_plus[i] += epsilon
                    scores_plus = X_q @ w_plus
                    ndcg_plus = self._ndcg_at_k(scores_plus, y_q, self.k)

                    grad[i] += -(ndcg_plus - ndcg) / epsilon  # Negative because we minimize

            # Average gradient
            grad /= len(query_groups)

            # Gradient step
            self.weights_ -= self.learning_rate * grad

            # Proximal step for L1
            self.weights_ = np.sign(self.weights_) * np.maximum(
                np.abs(self.weights_) - self.learning_rate * self.lambda_l1, 0
            )

            # Track best weights
            avg_ndcg = total_ndcg / len(query_groups)
            if avg_ndcg > best_ndcg:
                best_ndcg = avg_ndcg
                best_weights = self.weights_.copy()

            self._print_progress(iteration, self.max_iter, {'ndcg': avg_ndcg})

        # Use best weights found
        self.weights_ = best_weights
        return self

    def predict_proba(self, X):
        """Predict scores with temperature-scaled softmax features."""
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X_softmax = self._apply_temperature_softmax(X)
        scores = X_softmax @ self.weights_
        return 1 / (1 + np.exp(-scores))


class ApproxNDCGFastTrainer(BaseTrainer):
    """
    Fast ApproxNDCG trainer with analytical gradients.

    Uses softrank approach for differentiable ranking approximation:
        approx_rank_i = 1 + sum_j sigmoid((s_j - s_i) / sigma)

    This allows analytical gradient computation instead of slow finite differences.
    Much faster than ApproxNDCGTrainer for high-dimensional features.
    """

    def __init__(self, lambda_l1=0.01, max_iter=1000, random_state=42,
                 learning_rate=0.1, temperature=1.0, k=10, sigma=1.0, verbose=False):
        """
        Initialize fast ApproxNDCG trainer.

        Args:
            lambda_l1: L1 regularization strength
            max_iter: Maximum iterations
            random_state: Random seed
            learning_rate: Learning rate
            temperature: Temperature for softmax over features
            k: Compute NDCG@k
            sigma: Temperature for softrank approximation (lower = sharper ranks)
            verbose: Whether to print progress during training
        """
        super().__init__(lambda_l1, temperature, max_iter, random_state, verbose)
        self.learning_rate = learning_rate
        self.k = k
        self.sigma = sigma

    @property
    def name(self) -> str:
        return "approx_ndcg_fast"

    def _compute_approx_ranks(self, scores):
        """
        Compute differentiable approximate ranks using sigmoid.

        approx_rank_i = 1 + sum_{j != i} sigmoid((s_j - s_i) / sigma)

        Returns:
            approx_ranks: (n_docs,) approximate ranks (1-indexed)
            sigmoid_diff: (n, n) sigmoid matrix for gradient computation
        """
        n = len(scores)
        # Compute pairwise score differences: diff[i,j] = s_j - s_i
        diff = scores.reshape(1, -1) - scores.reshape(-1, 1)  # (n, n)
        # Sigmoid of differences (how likely j ranks above i)
        sigmoid_diff = 1.0 / (1.0 + np.exp(-diff / self.sigma))
        # Set diagonal to 0 (don't compare with self)
        np.fill_diagonal(sigmoid_diff, 0)
        # Approximate rank = 1 + sum of sigmoids
        approx_ranks = 1.0 + sigmoid_diff.sum(axis=1)
        return approx_ranks, sigmoid_diff

    def _compute_approx_ndcg_and_grad(self, X_q, scores, labels, k):
        """
        Compute differentiable approximation of NDCG@k and its gradient.

        Uses soft position weights based on approximate ranks.
        Returns both NDCG value and gradient w.r.t. weights.
        """
        n = len(scores)
        approx_ranks, sigmoid_diff = self._compute_approx_ranks(scores)

        # Compute soft discounts: 1 / log2(approx_rank + 1)
        discounts = 1.0 / np.log2(approx_ranks + 1)

        # Soft top-k mask using sigmoid
        topk_weights = 1.0 / (1.0 + np.exp(-(k + 0.5 - approx_ranks) / self.sigma))

        # Approximate DCG
        dcg = np.sum(labels * discounts * topk_weights)

        # IDCG (ideal ranking - use hard sorting)
        ideal_labels = np.sort(labels)[::-1][:k]
        positions = np.arange(1, len(ideal_labels) + 1)
        idcg = np.sum(ideal_labels / np.log2(positions + 1))

        if idcg < 1e-10:
            return 0.0, np.zeros(X_q.shape[1])

        ndcg = dcg / idcg

        # Compute gradient analytically
        # Derivative of sigmoid: sigmoid' = sigmoid * (1 - sigmoid)
        sigmoid_deriv = sigmoid_diff * (1 - sigmoid_diff) / self.sigma

        # d(discount_i)/d(approx_rank_i)
        discount_deriv = -1.0 / ((approx_ranks + 1) * np.log(2) * np.log2(approx_ranks + 1)**2)

        # d(topk_weight_i)/d(approx_rank_i)
        topk_deriv = -topk_weights * (1 - topk_weights) / self.sigma

        # Gradient of DCG w.r.t. scores using vectorized computation
        # d(rank_i)/d(s_k) for k != i: sigmoid_deriv[i, k]
        # d(rank_i)/d(s_i): -sum_j sigmoid_deriv[i, j]

        # Coefficient for each document in DCG gradient
        coef = labels * (discount_deriv * topk_weights + discounts * topk_deriv)

        # d(DCG)/d(s_k) = sum_i coef_i * d(rank_i)/d(s_k)
        grad_scores = np.zeros(n)
        for k_idx in range(n):
            # Contribution from d(rank_i)/d(s_k) for i != k
            grad_scores[k_idx] = np.sum(coef * sigmoid_deriv[:, k_idx])
            # Contribution from d(rank_k)/d(s_k) = -sum_j sigmoid_deriv[k, j]
            grad_scores[k_idx] += coef[k_idx] * (-np.sum(sigmoid_deriv[k_idx, :]))

        # Gradient of NDCG = gradient of DCG / IDCG
        grad_scores /= idcg

        # Convert to gradient w.r.t. weights: dL/dw = -sum_i (dNDCG/ds_i) * x_i
        # Negative because we minimize -NDCG
        grad_weights = -X_q.T @ grad_scores

        return ndcg, grad_weights

    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit model by maximizing approx NDCG with analytical gradients.

        Args:
            X_train: Training features (n_docs, n_features)
            y_train: Training labels (n_docs,)
            docs_per_query_train: Array of docs per query (required)
        """
        if docs_per_query_train is None:
            raise ValueError("ApproxNDCGFast trainer requires docs_per_query_train")

        n_features = X_train.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Initialize weights
        self.weights_ = rng.randn(n_features) * 0.01
        self.intercept_ = 0.0

        # Apply temperature-scaled softmax to features
        X_train_softmax = self._apply_temperature_softmax(X_train)

        # Group documents by query
        query_groups = []
        doc_offset = 0
        for n_docs in docs_per_query_train:
            query_groups.append({
                'X': X_train_softmax[doc_offset:doc_offset + n_docs],
                'y': y_train[doc_offset:doc_offset + n_docs]
            })
            doc_offset += n_docs

        # Gradient descent
        best_weights = self.weights_.copy()
        best_ndcg = -np.inf

        for iteration in range(self.max_iter):
            grad = np.zeros(n_features)
            total_ndcg = 0.0

            for group in query_groups:
                X_q = group['X']
                y_q = group['y']

                # Compute scores
                scores = X_q @ self.weights_

                # Compute approximate NDCG and gradient together
                ndcg, grad_q = self._compute_approx_ndcg_and_grad(X_q, scores, y_q, self.k)
                total_ndcg += ndcg
                grad += grad_q

            # Average gradient
            grad /= len(query_groups)

            # Gradient step
            self.weights_ -= self.learning_rate * grad

            # Proximal step for L1
            self.weights_ = np.sign(self.weights_) * np.maximum(
                np.abs(self.weights_) - self.learning_rate * self.lambda_l1, 0
            )

            # Track best weights
            avg_ndcg = total_ndcg / len(query_groups)
            if avg_ndcg > best_ndcg:
                best_ndcg = avg_ndcg
                best_weights = self.weights_.copy()

            self._print_progress(iteration, self.max_iter, {'ndcg': avg_ndcg})

        # Use best weights found
        self.weights_ = best_weights
        return self

    def predict_proba(self, X):
        """Predict scores with temperature-scaled softmax features."""
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X_softmax = self._apply_temperature_softmax(X)
        scores = X_softmax @ self.weights_
        return 1 / (1 + np.exp(-scores))


class GroupLassoTrainer(BaseTrainer):
    """
    Group Lasso trainer for layer-wise sparse head selection.

    Optimizes:
        L = (1/n) * sum(BCE) + lambda * sum(||w_layer||_2)

    Encourages entire layers to be selected or deselected together.
    Discovers which layers are most important for retrieval.
    """

    def __init__(self, lambda_l1=0.01, temperature=1.0, max_iter=1000, random_state=42,
                 num_layers=32, num_heads_per_layer=32, class_weight='balanced', verbose=False):
        """
        Initialize Group Lasso trainer.

        Args:
            lambda_l1: Group lasso regularization strength
            temperature: Temperature for softmax over features. Default: 1.0 (standard softmax)
            max_iter: Maximum iterations
            random_state: Random seed
            num_layers: Number of layers in the model
            num_heads_per_layer: Number of attention heads per layer
            class_weight: Class weighting for BCE
            verbose: Whether to print progress during training
        """
        super().__init__(lambda_l1, temperature, max_iter, random_state, verbose)
        self.num_layers = num_layers
        self.num_heads_per_layer = num_heads_per_layer
        self.class_weight = class_weight

    @property
    def name(self) -> str:
        return "group_lasso"

    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit model using Group Lasso.

        Uses proximal gradient descent with group-wise soft thresholding.
        """
        n_features = X_train.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Initialize weights
        self.weights_ = rng.randn(n_features) * 0.01
        self.intercept_ = 0.0

        # Compute class weights if needed
        if self.class_weight == 'balanced':
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            w_pos = len(y_train) / (2 * n_pos) if n_pos > 0 else 1.0
            w_neg = len(y_train) / (2 * n_neg) if n_neg > 0 else 1.0
            sample_weights = np.where(y_train == 1, w_pos, w_neg)
        else:
            sample_weights = np.ones(len(y_train))

        learning_rate = 0.01

        # Apply temperature-scaled softmax to features
        X_train_softmax = self._apply_temperature_softmax(X_train)

        # Proximal gradient descent
        for iteration in range(self.max_iter):
            # Compute predictions
            scores = X_train_softmax @ self.weights_ + self.intercept_
            probs = 1 / (1 + np.exp(-scores))

            # Compute loss for progress
            eps = 1e-15
            loss = -np.mean(sample_weights * (y_train * np.log(probs + eps) + (1 - y_train) * np.log(1 - probs + eps)))

            self._print_progress(iteration, self.max_iter, {'loss': loss})

            # BCE gradient
            grad = X_train_softmax.T @ ((probs - y_train) * sample_weights) / len(y_train)

            # Gradient step
            self.weights_ -= learning_rate * grad

            # Group-wise proximal step (block soft thresholding)
            for layer in range(self.num_layers):
                start_idx = layer * self.num_heads_per_layer
                end_idx = start_idx + self.num_heads_per_layer

                if end_idx <= n_features:
                    w_layer = self.weights_[start_idx:end_idx]
                    layer_norm = np.linalg.norm(w_layer)

                    if layer_norm > 0:
                        # Block soft thresholding
                        shrinkage = max(0, 1 - learning_rate * self.lambda_l1 / layer_norm)
                        self.weights_[start_idx:end_idx] = shrinkage * w_layer

        return self

    def predict_proba(self, X):
        """Predict probability of positive class with temperature-scaled softmax features."""
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X_softmax = self._apply_temperature_softmax(X)
        scores = X_softmax @ self.weights_ + self.intercept_
        return 1 / (1 + np.exp(-scores))

    def get_layer_importance(self):
        """
        Get importance score for each layer.

        Returns:
            layer_importance: (num_layers,) array of L2 norms per layer
        """
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        layer_importance = []
        for layer in range(self.num_layers):
            start_idx = layer * self.num_heads_per_layer
            end_idx = start_idx + self.num_heads_per_layer

            if end_idx <= len(self.weights_):
                w_layer = self.weights_[start_idx:end_idx]
                layer_importance.append(np.linalg.norm(w_layer))
            else:
                break

        return np.array(layer_importance)


class RankNetTrainer(BaseTrainer):
    """
    RankNet trainer - pairwise ranking with smooth loss.

    Optimizes:
        L = (1/n_pairs) * sum(-log(sigmoid(s_pos - s_neg))) + lambda * ||w||_1

    Smooth pairwise ranking loss that's easier to optimize than hinge loss.
    """

    def __init__(self, lambda_l1=0.01, temperature=1.0, max_iter=1000, random_state=42,
                 learning_rate=0.01, verbose=False):
        """
        Initialize RankNet trainer.

        Args:
            lambda_l1: L1 regularization strength
            temperature: Temperature for softmax over features. Default: 1.0 (standard softmax)
            max_iter: Maximum iterations
            random_state: Random seed
            learning_rate: Learning rate
            verbose: Whether to print progress during training
        """
        super().__init__(lambda_l1, temperature, max_iter, random_state, verbose)
        self.learning_rate = learning_rate

    @property
    def name(self) -> str:
        return "ranknet"

    def fit(self, X_train, y_train, docs_per_query_train=None):
        """
        Fit model using RankNet pairwise loss.

        Args:
            X_train: Training features
            y_train: Training labels
            docs_per_query_train: Array of docs per query (required)
        """
        if docs_per_query_train is None:
            raise ValueError("RankNet trainer requires docs_per_query_train")

        n_features = X_train.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Initialize weights
        self.weights_ = rng.randn(n_features) * 0.01
        self.intercept_ = 0.0

        # Apply temperature-scaled softmax to features
        X_train_softmax = self._apply_temperature_softmax(X_train)

        # Create pairwise training examples
        pairs = []
        doc_offset = 0
        for n_docs in docs_per_query_train:
            X_q = X_train_softmax[doc_offset:doc_offset + n_docs]
            y_q = y_train[doc_offset:doc_offset + n_docs]

            pos_indices = np.where(y_q == 1)[0]
            neg_indices = np.where(y_q == 0)[0]

            # Create all positive-negative pairs
            for pos_idx in pos_indices:
                for neg_idx in neg_indices:
                    pairs.append({
                        'x_pos': X_q[pos_idx],
                        'x_neg': X_q[neg_idx]
                    })

            doc_offset += n_docs

        if len(pairs) == 0:
            raise ValueError("No positive-negative pairs found")

        # Gradient descent
        for iteration in range(self.max_iter):
            grad = np.zeros(n_features)
            loss = 0.0

            for pair in pairs:
                x_pos = pair['x_pos']
                x_neg = pair['x_neg']

                # Score difference
                s_diff = (x_pos - x_neg) @ self.weights_

                # Sigmoid
                sigmoid = 1 / (1 + np.exp(-s_diff))

                # Loss: -log(sigmoid(s_pos - s_neg))
                loss += -np.log(sigmoid + 1e-15)

                # Gradient: -(1 - sigmoid) * (x_pos - x_neg)
                grad += -(1 - sigmoid) * (x_pos - x_neg)

            # Average over pairs
            grad /= len(pairs)
            loss /= len(pairs)

            self._print_progress(iteration, self.max_iter, {'loss': loss})

            # Gradient step
            self.weights_ -= self.learning_rate * grad

            # Proximal step for L1
            self.weights_ = np.sign(self.weights_) * np.maximum(
                np.abs(self.weights_) - self.learning_rate * self.lambda_l1, 0
            )

        return self

    def predict_proba(self, X):
        """Predict scores with temperature-scaled softmax features."""
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X_softmax = self._apply_temperature_softmax(X)
        scores = X_softmax @ self.weights_
        return 1 / (1 + np.exp(-scores))


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
                 learning_rate=0.01, temperature=1.0, verbose=False):
        """
        Initialize InfoNCE trainer.

        Args:
            lambda_l1: L1 regularization strength
            max_iter: Maximum iterations for optimization
            random_state: Random seed
            learning_rate: Learning rate for gradient descent
            temperature: Temperature for softmax scaling over features. Default: 1.0 (standard softmax)
            verbose: Whether to print progress during training
        """
        super().__init__(lambda_l1, temperature, max_iter, random_state, verbose)
        self.learning_rate = learning_rate

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

        # Apply temperature-scaled softmax to features (like CoRe)
        X_train_softmax = self._apply_temperature_softmax(X_train)

        # Group documents by query
        query_groups = []
        doc_offset = 0
        for n_docs in docs_per_query_train:
            query_groups.append({
                'X': X_train_softmax[doc_offset:doc_offset + n_docs],
                'y': y_train[doc_offset:doc_offset + n_docs]
            })
            doc_offset += n_docs

        # Proximal gradient descent
        for iteration in range(self.max_iter):
            grad = np.zeros(n_features)
            total_loss = 0.0

            for group in query_groups:
                X_q = group['X']
                y_q = group['y']

                # Compute scores (features already have temperature-scaled softmax applied)
                scores = X_q @ self.weights_

                # Softmax probabilities
                scores_exp = np.exp(scores - scores.max())  # Numerical stability
                probs = scores_exp / scores_exp.sum()

                # InfoNCE loss: -log(prob of positive)
                pos_mask = y_q == 1
                if pos_mask.any():
                    total_loss += -np.log(probs[pos_mask].sum() + 1e-15)

                # Gradient: sum over docs of (prob - target) * x
                # For InfoNCE, target is 1 for positive docs, 0 for negative
                targets = y_q / max(y_q.sum(), 1)  # Normalize targets
                grad += X_q.T @ (probs - targets)

            # Average gradient and loss
            grad /= len(query_groups)
            avg_loss = total_loss / len(query_groups)

            self._print_progress(iteration, self.max_iter, {'loss': avg_loss})

            # Gradient step
            self.weights_ -= self.learning_rate * grad

            # Proximal step (soft thresholding for L1)
            self.weights_ = np.sign(self.weights_) * np.maximum(
                np.abs(self.weights_) - self.learning_rate * self.lambda_l1, 0
            )

        return self

    def predict_proba(self, X):
        """Predict scores with temperature-scaled softmax features."""
        if self.weights_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X_softmax = self._apply_temperature_softmax(X)
        scores = X_softmax @ self.weights_
        # Convert to probabilities using sigmoid
        return 1 / (1 + np.exp(-scores))


# Registry of available trainers
TRAINERS = {
    'bce': BCETrainer,
    'bce_temp': BCEWithTemperatureTrainer,
    'infonce': InfoNCETrainer,
    'hinge': HingeLossTrainer,
    'approx_ndcg': ApproxNDCGTrainer,
    'approx_ndcg_fast': ApproxNDCGFastTrainer,
    'group_lasso': GroupLassoTrainer,
    'ranknet': RankNetTrainer,
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
