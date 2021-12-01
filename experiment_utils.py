from utils import *
from preprocessing import preprocess
from windowfy import windowfy
from featurizing import featurize
from tfidf_featurizer import combine_features, tfidf_featurize
from training import train, do_ensemble, do_train, evaluate_dl_time
from training_traditional import train_and_evaluate
from training_traditional import evaluate as evaluate_trad_time
from eval_erisk import evaluate, ensemble_vote
from IPython.display import display, Markdown
from itertools import product
import tensorflow
import numpy as np
import time

dl_model_names = ["cnn_model", "lstm_model_32", "lstm_model_16", "lstm_model"]
trad_model_names = ["svm", "bayes"]
      
def traverse(d):
    K,V = zip(*d.items())
    for v in product(*(v if isinstance(v,list) else traverse(v) for v in V)):
        yield dict(zip(K,v))

class Experiment():
    
    def __init__(self, models, ensemble_combinations, eval_filename, random_seed=42, name=None):
        self.models = models
        self.ensemble_combinations = ensemble_combinations
        self.eval_filename = eval_filename
        if name is None:
            self.name = time.process_time()
        else:
            self.name = name
        self.seed = random_seed
        self.set_seed(random_seed)
        
    
    def prepare_data(self, params):
        logger("PREPARING DATA FOR PARAMS {}".format(params))
        self.train_users, self.y_train, self.test_users, self.y_test, self.train_samples, self.X_train, self.X_test = windowfy(window_size=params["feat_window_size"], max_size=params["max_size"], sample_weights_size=params["sample_weights_size"], is_oversample=params["oversample"], include_new_data=params["include_new_data"], sampling_strategy=params["sampling_strategy"], random_state=self.seed)
        self.feats_train, self.feats_test = featurize(calculate_feats=True, 
                                            include_feats=params["include_feats"],
                                            train_users=self.train_users, test_users=self.test_users,
                                            discretize=params["discretize"], 
                                            discretize_size=params["discretize_size"],
                                            dis_strategy=params["dis_strategy"], 
                                            normalize=params["normalize"],
                                            scale=params["scale"])
        self.tfidf_train, self.tfidf_test = tfidf_featurize(self.train_users, self.test_users, max_features=params["tfidf_max_features"])

        self.feats_train_comb, self.feats_test_comb = combine_features([self.tfidf_train, self.feats_train], [self.tfidf_test, self.feats_test])

        self.feats_train_comb = self.feats_train_comb.toarray()
        self.feats_test_comb = self.feats_test_comb.toarray() 
        
    def train_and_evaluate_model(self, params, weights_combinations=None):
        self.y_preds = {}
        params["weights"] = None
        for model_name in self.models:
            params["model"] = model_name
            if model_name in trad_model_names:
                logger("TRAINING AND EVALUATING TRADITIONAL MODEL {}".format(model_name))
                y_pred, classifier = train_and_evaluate(self.feats_train_comb, self.y_train, self.feats_test_comb, self.y_test, self.train_samples, classifier_name=model_name, strategy="weights")
                t = time.process_time()
                logger("Evaluating after getting time {}".format(t))
                evaluate_trad_time(classifier, self.feats_test_comb, self.y_test)
                elapsed_time = time.process_time() - t
                logger("Evaluated with elapsed time {}".format(elapsed_time))
            else:
                logger("TRAINING AND EVALUATING DL MODEL {}".format(model_name))
                y_pred = self.iterate_dl_model(params)
                logger("Evaluating for elapsed time")
                elapsed_time = evaluate_dl_time(model_name=params["model"], maxlen=params["maxlen"], epochs=params["epochs"],
                              batch_size=params["batch_size"],
                              shuffle=params["shuffle"], patience=params["patience"], 
                              feats_train=self.feats_train, feats_test=self.feats_test, 
                              X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test, 
                              train_sample_weights=self.train_samples, name=self.name)
                logger("Evaluated with elapsed time {}".format(elapsed_time))
            logger("EVALUATING FOR WINDOW SIZES 1, 2 AND 3 MODEL {}".format(model_name))
            params["eval_time"] = elapsed_time
            params["eval_window_size"] = 1
            eval_resul = evaluate(1, 10, params, y_pred=y_pred, test_users=self.test_users, resuls_file=self.eval_filename)
            params["eval_window_size"] = 2
            eval_resul = evaluate(2, 10, params, y_pred=y_pred, test_users=self.test_users, resuls_file=self.eval_filename)
            params["eval_window_size"] = 3
            eval_resul = evaluate(3, 10, params, y_pred=y_pred, test_users=self.test_users, resuls_file=self.eval_filename)
            
            self.y_preds[model_name] = y_pred

        for ensemble_ver in self.ensemble_combinations:
            if weights_combinations is None:
                weights_combinations = [[1,1,1]]
            
            for weights in weights_combinations:
                logger("EVALUATING ENSEMBLE {} with weights {}".format(ensemble_ver, weights))
                ensemble_preds = [self.y_preds[model_name] for model_name in ensemble_ver]
                ensemble_preds = np.array(ensemble_preds)
                y_pred = ensemble_vote(ensemble_preds, weights)

                params["model"] = ensemble_ver
                params["weights"] = weights

                logger("EVALUATING ENSEMBLE {} WITH WEIGHTS {} FOR WINDOW SIZES 1, 2 AND 3".format(ensemble_ver, weights))
                params["eval_window_size"] = 1
                eval_resul = evaluate(1, 10, params, y_pred=y_pred, test_users=self.test_users, resuls_file=self.eval_filename)
                params["eval_window_size"] = 2
                eval_resul = evaluate(2, 10, params, y_pred=y_pred, test_users=self.test_users, resuls_file=self.eval_filename)
                params["eval_window_size"] = 3
                eval_resul = evaluate(3, 10, params, y_pred=y_pred, test_users=self.test_users, resuls_file=self.eval_filename)   
    
    
    def iterate_dl_model(self, params):
    
        model_resuls = {}
        iterations = params["iterations"]
        logger("STARTING ITERATION FOR DL MODEL {} FOR {} ITERATIONS".format(params["model"], params["iterations"]))
        for i in range(0, iterations):
            y_pred = do_train(model_name=params["model"], maxlen=params["maxlen"], epochs=params["epochs"],
                              batch_size=params["batch_size"],
                              shuffle=params["shuffle"], patience=params["patience"], 
                              feats_train=self.feats_train, feats_test=self.feats_test, 
                              X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test, 
                              train_sample_weights=self.train_samples, name=self.name)
            eval_resul = evaluate(1, 10, params, y_pred=y_pred, test_users=self.test_users, save=False)
            model_resuls[eval_resul['latency_weighted_f1']] = y_pred.flatten()

        return model_resuls[max(model_resuls.keys())]
    
    def set_seed(self, seed_num):
        np.random.seed(seed_num)
        tensorflow.random.set_seed(seed_num)
        logger("Initialized numpy random and tensorflow random seed at {}".format(seed_num))