from utils import eval_performance
import csv
import os
from utils import load_pickle
import filenames as fp

import numpy as np
from utils import load_parameters
from utils import *
from datetime import datetime
import subprocess

train_g_truth_file = "/datos/erisk/deep-learning/data/erisk2021_training_data/golden_truth.txt"
test_g_truth_file = "/datos/erisk/deep-learning/data/erisk2021_test_data/golden_truth.txt"


def evaluate(feats_window_size=10, window_size=100):
    
    g_truth = load_golden_truth(test_g_truth_file, test_collection=True)

    test_resuls = load_pickle(pickle_path, fp.resul_file)
    test_scores = load_pickle(pickle_path, fp.score_file)
    
    X_test = load_pickle(pickle_path, "X_test.pkl")

    user_resul = prepare_data(X_test, test_resuls)
    user_scores = prepare_data(X_test, test_scores)

    test_resul_proc = process_decisions_w2(user_resul, user_scores, feats_window_size, max_strategy=window_size)
    eval_resuls = eval_performance(test_resul_proc, g_truth)

    logger(eval_resuls)
    return eval_resuls





def write_csv(eval_resuls):

    data = {}
    data["commit hash"] = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    data["timestamp"] = dt_string


    params = load_parameters()
    data.update(params)
    data.update(eval_resuls)

    erisk_eval_file = os.path.join(fp.resuls_path, fp.erisk_eval_filename)
    csv_file = erisk_eval_file

    csv_columns = data.keys()
    dict_data = [data]

    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            if os.path.getsize(csv_file) == 0:
                writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    csv_columns = eval_resuls.keys()
    dict_data = [eval_resuls]





def prepare_data(test_x, resul_array):

    test_users = np.array(test_x[["user"]]).flatten()
    resul_array = resul_array.tolist()
    test_users = test_users.tolist()

    user_tuples = list(zip(test_users, resul_array))
    user_dict = array_to_dict(user_tuples)

    return user_dict



def array_to_dict(l):
    d = dict()
    [d[t[0]].append(t[1]) if t[0] in list(d.keys())
     else d.update({t[0]: [t[1]]}) for t in l]
    return d

def process_decisions_w2(user_decisions, user_scores, feats_window_size, max_strategy=5):
    decision_list = []
    new_user_decisions = {}
    new_user_sequence = {}
    max = max_strategy

    for user, decisions in user_decisions.items():
        new_user_decisions[user] = []
        new_user_sequence[user] = []

    # politica de decisiones: decidimos que un usuario es positivo a partir del 5 mensaje positivo consecutivo
    # a partir de ahi, todas las decisiones deben ser positivas, y la secuencia mantenerse estable
    for user, decisions in user_decisions.items():
        count = 0
        for i in range(0, len(decisions)):
            if decisions[i] == 0 and count < max:
                count = 0
                new_user_decisions[user].append(0)
                new_user_sequence[user].append(i)
            elif decisions[i] == 1 and count < max:
                count = count + 1
                new_user_decisions[user].append(0)
                new_user_sequence[user].append(i)
            elif count >= max:
                new_user_decisions[user].append(1)
                new_user_sequence[user].append(new_user_sequence[user][i - 1])

    # lo montamos en el formato que acepta el evaluador
    for user, decisions in new_user_decisions.items():
        decision_list.append(
            {"nick": user, "decision": new_user_decisions[user][-1], "sequence": new_user_sequence[user][-1], "score":
                user_scores[user][-1]})

    return decision_list


def process_decisions_w1(user_decisions, user_scores, feat_window_size, max_strategy=5):
    decision_list = []
    new_user_decisions = {}
    new_user_sequence = {}
    max = max_strategy

    for user, decisions in user_decisions.items():
        new_user_decisions[user] = []
        new_user_sequence[user] = []

    # politica de decisiones: decidimos que un usuario es positivo a partir del 5 mensaje positivo consecutivo
    # a partir de ahi, todas las decisiones deben ser positivas, y la secuencia mantenerse estable
    for user, decisions in user_decisions.items():
        count = 0
        for i in range(0, len(decisions)):
            if decisions[i] == 0 and count < max:
                count = 0
                new_user_decisions[user].append(0)
                new_user_sequence[user].append(i+feat_window_size)
            elif decisions[i] == 1 and count < max:
                count = count +1
                new_user_decisions[user].append(0)
                new_user_sequence[user].append(i+feat_window_size)
            elif count >= max:
                new_user_decisions[user].append(1)
                new_user_sequence[user].append(new_user_sequence[user][i-1])

    # lo montamos en el formato que acepta el evaluador
    for user, decisions in new_user_decisions.items():
        decision_list.append(
            {"nick": user, "decision": new_user_decisions[user][-1], "sequence": new_user_sequence[user][-1], "score":
                user_scores[user][-1]})

    return decision_list

def load_golden_truth(filename):
    g_path = filename
    g_truth = {line.split()[0]: int(line.split()[1]) for line in open(g_path)}
    return g_truth

if __name__ == '__main__':
    main()
    
    
def penalty(delay):
    import numpy as np
    p = 0.0078
    pen = -1.0 + 2.0 / (1 + np.exp(-p * (delay - 1)))
    return (pen)


def eval_performance(run_results, qrels):
    import numpy as np

    total_pos = n_pos(qrels)

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    erdes5 = np.zeros(len(run_results))
    erdes50 = np.zeros(len(run_results))
    ierdes = 0
    latency_tps = list()
    penalty_tps = list()

    for r in run_results:
        try:
            # print(qrels[ r['nick']   ], r['decision'], r['nick'], qrels[ r['nick']   ] ==  r['decision'] )
            if (qrels[r['nick']] == r['decision']):
                if (r['decision'] == 1):
                    # print('dec = 1')
                    true_pos += 1
                    erdes5[ierdes] = 1.0 - (1.0 / (1.0 + np.exp((r['sequence'] + 1) - 5.0)))
                    erdes50[ierdes] = 1.0 - (1.0 / (1.0 + np.exp((r['sequence'] + 1) - 50.0)))
                    latency_tps.append(r['sequence'] + 1)
                    penalty_tps.append(penalty(r['sequence'] + 1))
                else:
                    # print('dec = 0')
                    true_neg += 1
                    erdes5[ierdes] = 0
                    erdes50[ierdes] = 0
            else:
                if (r['decision'] == 1):
                    # print('++')
                    false_pos += 1
                    erdes5[ierdes] = float(total_pos) / float(len(qrels))
                    erdes50[ierdes] = float(total_pos) / float(len(qrels))
                else:
                    # print('****')
                    false_neg += 1
                    erdes5[ierdes] = 1
                    erdes50[ierdes] = 1

        except KeyError:
            print("User does not appear in the qrels:" + r['nick'])

        ierdes += 1

    if (true_pos == 0):
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = float(true_pos) / float(true_pos + false_pos)
        recall = float(true_pos) / float(total_pos)
        F1 = 2 * (precision * recall) / (precision + recall)

    speed = 1 - np.median(np.array(penalty_tps))

    eval_results = {}
    eval_results['precision'] = precision
    eval_results['recall'] = recall
    eval_results['F1'] = F1
    eval_results['ERDE_5'] = np.mean(erdes5)
    eval_results['ERDE_50'] = np.mean(erdes50)
    eval_results['median_latency_tps'] = np.median(np.array(latency_tps))
    eval_results['median_penalty_tps'] = np.median(np.array(penalty_tps))
    eval_results['speed'] = speed
    eval_results['latency_weighted_f1'] = F1 * speed

    return eval_results


def n_pos(qrels):
    total_pos = 0
    for key in qrels:
        total_pos += qrels[key]
    return (total_pos)