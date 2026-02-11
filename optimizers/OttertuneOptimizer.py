import numpy as np
import random
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from utils import DistanceUtil


class OtterTuneOptimizer(BaseOptimizer):
    """
    OtterTune-style optimizer, adapted to your tabular framework.

    GUARANTEES (framework safety):
    - We NEVER modify the raw/snapped hp dict format that model_config expects.
    - nearestRow() is ALWAYS called on raw/snapped values only.
    - Scaling is used ONLY inside the GP surrogate (private numeric vectors).
    - Cache keys remain snapped row tuples in your original column order.

    Components:
    - Multi-objective scores: model_wrapper.get_score(valid_hp) -> tuple (maximize each)
    - Scalarized fitness: D2H to ideal=[1,...,1] (minimize)
    - Lasso-path knob ranking (OtterTune-ish)
    - GP surrogate + Expected Improvement over a candidate pool (tabular-friendly)
    - Budget: counts UNIQUE snapped rows only, stops at n_trials
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        self.config_space, _, _ = self.model_config.get_configspace()

        # cache: row_tuple -> (scores_tuple, d2h_float)
        self.cache = {}

        # number of objectives from score tuple
        self.num_objectives = len(
            self.model_wrapper.get_score(
                {c: self.X_df.iloc[0][c] for c in self.columns}
            )
        )
        self.ideal = [1.0] * self.num_objectives

        # knob names (what we rank with lasso)
        self.knob_names = list(self.columns)

        # rng
        self._rng = np.random.RandomState(seed)

        self.best_config = None
        self.best_value = None  # best D2H (lower is better)

    # -----------------------------
    # Utils (match your style)
    # -----------------------------
    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        # IMPORTANT: do NOT scale or transform this input (framework contract)
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    # -----------------------------
    # Score + D2H evaluation
    # -----------------------------
    def _eval_scores_and_d2h(self, valid_hp):
        """
        Returns (scores_tuple, d2h_float). Caches by snapped row.
        """
        key = self._row_tuple(valid_hp)
        if key in self.cache:
            return self.cache[key]

        try:
            scores = self.model_wrapper.get_score(valid_hp)  # tuple of objectives (maximize)
        except Exception:
            scores = tuple(0.0 for _ in range(self.num_objectives))

        d2h = DistanceUtil.d2h(self.ideal, scores)

        self.cache[key] = (scores, float(d2h))
        return self.cache[key]

    # -----------------------------
    # Lasso-path knob ranking (OtterTune-ish)
    # -----------------------------
    def _lasso_rank_knobs(self, X, y, knob_names):
        """
        Rank knobs by when they first become non-zero along lasso_path.
        X: (n, d) numeric matrix
        y: (n,) numeric target (we'll use d2h as target)
        """
        xs = StandardScaler().fit_transform(X)
        ys = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

        # coef_path shape: (d, n_alphas)
        _, coef_path, _ = lasso_path(xs, ys)

        first_nonzero = np.full(len(knob_names), fill_value=np.inf)
        for j in range(coef_path.shape[0]):
            nz = np.where(np.abs(coef_path[j, :]) > 1e-12)[0]
            if len(nz) > 0:
                first_nonzero[j] = nz[0]

        order = np.argsort(first_nonzero)
        ranked = [knob_names[i] for i in order if first_nonzero[i] != np.inf]
        ranked += [knob_names[i] for i in range(len(knob_names)) if first_nonzero[i] == np.inf]
        return ranked

    # -----------------------------
    # EI for minimization (no scipy)
    # -----------------------------
    def _norm_pdf(self, z):
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)

    def _norm_cdf(self, z):
        # Abramowitz-Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
        d = 0.3989423 * np.exp(-z * z / 2.0)
        prob = d * t * (
            0.3193815
            + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))
        )
        return np.where(z >= 0, 1.0 - prob, prob)

    def _ei_min(self, mu, sigma, best, xi=0.0):
        sigma = np.maximum(sigma, 1e-12)
        imp = best - mu - xi
        Z = imp / sigma
        return imp * self._norm_cdf(Z) + sigma * self._norm_pdf(Z)

    # -----------------------------
    # ConfigSpace sampling (same hyperparam handling as NSGAIII)
    # -----------------------------
    def _sample_raw_hp(self):
        """
        Samples one raw configuration by iterating ConfigSpace hyperparameters,
        matching the NSGAIII style (Constant/Ordinal/Categorical).
        """
        raw_hp = {}
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                raw_hp[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                raw_hp[hp.name] = random.choice(list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                raw_hp[hp.name] = random.choice(list(hp.choices))
            else:
                raise ValueError("Unsupported hyperparameter")
        return raw_hp

    def _sample_unique_snapped(self, evaluated_keys, k, max_tries=20000):
        """
        Return up to k snapped hp dicts whose row_tuple is not in evaluated_keys.
        evaluated_keys is the global budget set (UNIQUE snapped rows).
        """
        out = []
        tries = 0
        while len(out) < k and tries < max_tries:
            tries += 1
            raw = self._sample_raw_hp()
            valid = self._nearest_row(raw)   # IMPORTANT: preserve framework format
            key = self._row_tuple(valid)
            if key in evaluated_keys:
                continue
            out.append(valid)
            evaluated_keys.add(key)
        return out

    # -----------------------------
    # Main optimize loop (budget-respecting)
    # -----------------------------
    def optimize(self):
        n_trials = int(self.config["n_trials"])

        # Tunables (safe defaults)
        lasso_subsample = int(self.config.get("lasso_subsample", 512))
        top_k = int(self.config.get("ottertune_top_k_knobs", max(2, len(self.knob_names))))
        n_init = int(self.config.get("n_init", 11))
        candidate_pool = int(self.config.get("candidate_pool", 256))
        gp_alpha = float(self.config.get("gp_alpha", 1e-4))   # IMPORTANT: default slightly larger for stability
        ei_xi = float(self.config.get("ei_xi", 0.0))
        gp_restarts = int(self.config.get("gp_restarts", 3))

        n_init = min(n_init, n_trials)

        # -----------------------------
        # 1) Lasso ranking using table subsample
        # -----------------------------
        # X_df is already numeric/encoded for nearestRow. We will NOT change it.
        X_all = self.X_df[self.columns].values
        sub_n = min(lasso_subsample, len(X_all))
        idx = self._rng.choice(len(X_all), size=sub_n, replace=False)
        X_sub = X_all[idx]

        # Evaluate D2H for those rows (snapped to table anyway)
        y_sub = []
        for row in X_sub:
            hp = {c: self._clean(v) for c, v in zip(self.columns, row)}
            hp = self._nearest_row(hp)
            _, d2h = self._eval_scores_and_d2h(hp)
            y_sub.append(d2h)
        y_sub = np.array(y_sub, dtype=float)

        ranked = self._lasso_rank_knobs(X_sub, y_sub, self.knob_names)
        top_k = min(max(2, top_k), len(ranked))
        use_knobs = ranked[:top_k]

        # GP-only vectorization (does NOT affect framework dict format)
        def hp_to_vec(hp):
            return np.array([float(hp[k]) for k in use_knobs], dtype=float)

        # -----------------------------
        # 2) Init design: unique snapped rows
        # -----------------------------
        evaluated_keys = set()
        init_hps = self._sample_unique_snapped(evaluated_keys, n_init)

        X_train_raw, y_train = [], []
        best_d2h = float("inf")

        for hp in init_hps:
            _, d2h = self._eval_scores_and_d2h(hp)
            X_train_raw.append(hp_to_vec(hp))
            y_train.append(d2h)
            if d2h < best_d2h:
                best_d2h = d2h

        X_train_raw = np.vstack(X_train_raw)
        y_train = np.array(y_train, dtype=float)

        # GP input scaling ONLY (framework safety)
        gp_x_scaler = StandardScaler()
        X_train = gp_x_scaler.fit_transform(X_train_raw)

        # -----------------------------
        # 3) BO loop until budget reached (UNIQUE evals)
        # -----------------------------
        while len(evaluated_keys) < n_trials:
            # GP kernel with explicit noise term
            # (helps L-BFGS convergence, especially with snapping/duplicates)
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(
                    length_scale=np.ones(len(use_knobs)),
                    length_scale_bounds=(1e-2, 1e2),
                    nu=2.5,
                )
                + WhiteKernel(
                    noise_level=max(gp_alpha, 1e-12),
                    noise_level_bounds=(1e-12, 1e1),
                )
            )

            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=self.seed,
                alpha=max(gp_alpha, 1e-12),         # IMPORTANT: use config
                n_restarts_optimizer=max(0, gp_restarts),
            )
            gp.fit(X_train, y_train)

            # Candidate pool must be unseen snapped rows (but don’t spend budget on them)
            temp_keys = set(evaluated_keys)
            candidates = self._sample_unique_snapped(temp_keys, candidate_pool)

            if not candidates:
                # table exhausted or too many duplicates due to snapping
                break

            X_cand_raw = np.vstack([hp_to_vec(hp) for hp in candidates])
            X_cand = gp_x_scaler.transform(X_cand_raw)

            mu, std = gp.predict(X_cand, return_std=True)
            ei = self._ei_min(mu, std, best=best_d2h, xi=ei_xi)

            pick = int(np.argmax(ei))
            hp_next = candidates[pick]
            key_next = self._row_tuple(hp_next)

            if key_next in evaluated_keys:
                continue

            evaluated_keys.add(key_next)

            _, d2h_next = self._eval_scores_and_d2h(hp_next)

            # Update training set (keep raw + scaled in sync)
            X_train_raw = np.vstack([X_train_raw, hp_to_vec(hp_next)])
            y_train = np.append(y_train, d2h_next)

            # Refit scaler for stability under growing set (cheap, robust)
            X_train = gp_x_scaler.fit_transform(X_train_raw)

            if d2h_next < best_d2h:
                best_d2h = d2h_next

        # -----------------------------
        # 4) Incumbent (best D2H observed)
        # -----------------------------
        best_key = min(self.cache, key=lambda k: self.cache[k][1])
        best_hp = {c: best_key[i] for i, c in enumerate(self.columns)}
        best_d2h = float(self.cache[best_key][1])

        self.best_config = best_hp
        self.best_value = best_d2h

        return best_hp, best_d2h
