from utils import *
from preprocessing import preprocess
from featurizing import featurize
from training import train

oversample_values = [True, False]
normalize_values = [True, False]
discretize_values = [True, False]
exclude_feats_values = [[], ["methods", "terms", "instruments", "reasons"],
                 ["sentiment", "methods", "terms", "instruments", "reasons"]]



logger("Preprocessing with oversampling = False")
preprocess(is_oversample=False)
featurize(calculate_feats=True)

for exclude_feats_value in exclude_feats_values:
    for normalize_value in normalize_values:
        for discretize_value in discretize_values:
            logger("New experiment")
            featurize(calculate_feats=False, normalize=normalize_value, discretize=discretize_value,
                      exclude_feats=exclude_feats_value)
            train()
            logger("Finished experiment")
            
logger("Preprocessing with oversampling = True")
preprocess(is_oversample=True)
featurize(calculate_feats=True)

for exclude_feats_value in exclude_feats_values:
    for normalize_value in normalize_values:
        for discretize_value in discretize_values:
            logger("New experiment")
            featurize(calculate_feats=False, normalize=normalize_value, discretize=discretize_value,
                      exclude_feats=exclude_feats_value)
            train()
            logger("Finished experiment")

logger("Finished experiments")

