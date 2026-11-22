import logging, time
logging.disable(logging.INFO)
from football_edge.config import load_config
from football_edge.data.loader import load_matches, needs_features
from football_edge.evaluation.folds import make_folds
from football_edge.evaluation.runner import walk_forward
from football_edge.models.registry import model_factory
cfg = load_config("configs/default.yaml")
params = cfg.models.get("dixon_coles", {})
m = load_matches(cfg, features=needs_features(params))
folds = make_folds(m, 2026, 2026, cfg.evaluation.min_train_matches)
t = time.time()
p = walk_forward(m, folds, model_factory("dixon_coles", params))
p.to_parquet("reports/evaluation/dixon_coles/predictions_2026.parquet")
print(len(folds), "weekly folds,", len(p), "matches,", round(time.time() - t), "s;", p.columns.tolist()[:20])
