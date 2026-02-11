# import os
# import sys
# import tempfile
# import pandas as pd

# from optimizers.base_optimizer import BaseOptimizer
# from utils import DistanceUtil
# import pandas as pd
# import tempfile
# import os
# # ------------------------------------------------------------
# # Import PromiseTune without modifying it
# # ------------------------------------------------------------
# PROJECT_ROOT = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..")
# )
# PROMISETUNE_CODE = os.path.join(PROJECT_ROOT, "PromiseTune", "Code")
# sys.path.insert(0, PROMISETUNE_CODE)

# from PromiseTune import PromiseTune, PromiseTuneConfig


# class PromiseTuneOptimizer(BaseOptimizer):
#     """
#     Runs PromiseTune unchanged by presenting it
#     a single-objective (d2h) CSV.
#     """

#     def __init__(self, config, model_wrapper, model_config, logging_util, seed):
#         super().__init__(config, model_wrapper, model_config, logging_util, seed)

#         self.seed = seed
#         self.columns = list(self.model_wrapper.X.columns)

#         # number of objectives
#         example_hp = {
#             c: self.model_wrapper.X.iloc[0][c]
#             for c in self.columns
#         }
#         self.num_objectives = len(self.model_wrapper.get_score(example_hp))

#         self.best_config = None
#         self.best_value = None

#     # ------------------------------------------------------------
#     # Build PromiseTune-compatible CSV (X + single d2h column)
#     # ------------------------------------------------------------
#     def _create_promisetune_csv(self):
#         X = self.model_wrapper.X.copy()
#         y = self.model_wrapper.y.copy()

#         def row_d2h(row):
#             scores = tuple(row.values)
#             ideal = [1] * len(scores)
#             return DistanceUtil.d2h(ideal, scores)

#         d2h_col = y.apply(row_d2h, axis=1)

#         # IMPORTANT: name must end with "-"
#         df = pd.concat([X, d2h_col.rename("d2h-")], axis=1)

#         tmp = tempfile.NamedTemporaryFile(
#             suffix=".csv", delete=False, mode="w"
#         )
#         df.to_csv(tmp.name, index=False)
#         tmp.close()

#         return tmp.name

#     def optimize(self):
#         original_csv = self.model_config.get_dataset_file()
#         df = pd.read_csv(original_csv)

#         # --------------------------------------------------
#         # Compute d2h ONCE for all rows
#         # --------------------------------------------------
#         d2h_vals = []
#         for _, row in self.model_wrapper.X.iterrows():
#             hp = dict(row)
#             scores = self.model_wrapper.get_score(hp)
#             d2h = DistanceUtil.d2h([1] * len(scores), scores)
#             d2h_vals.append(d2h)

#         # 🚨 THIS NAME IS THE KEY 🚨
#         df["d2h$<"] = d2h_vals

#         # --------------------------------------------------
#         # Unique, auto-deleted temp CSV
#         # --------------------------------------------------
#         with tempfile.TemporaryDirectory(prefix="promisetune_") as tmpdir:
#             tmp_csv = os.path.join(tmpdir, "pt.csv")
#             df.to_csv(tmp_csv, index=False)

#             pt_config = PromiseTuneConfig(
#                 initial_size=min(10, self.config["n_trials"]),
#                 budget=self.config["n_trials"],
#                 rule=True
#             )

#             best_decision, _ = PromiseTune(
#                 filename=tmp_csv,
#                 config=pt_config,
#                 seed=self.seed
#             )
#         print(best_decision)
#         #if best_decision is None:
#         #    raise RuntimeError("PromiseTune returned no solution")

#         #final_hp = dict(zip(self.columns, best_decision))
#         #scores = self.model_wrapper.get_score(final_hp)
#         #final_d2h = DistanceUtil.d2h([1] * len(scores), scores)

#         #self.best_config = final_hp
#         #self.best_value = final_d2h
#         return "", 0.0
#         #return final_hp, final_d2h