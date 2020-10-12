#!/usr/bin/env python
import argparse
from kaggler.data_io import load_data
import logging
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

from const import N_FOLD, N_CLASS, SEED


def train_predict(train_file, test_file, feature_map_file, predict_valid_file, predict_test_file,
                  feature_imp_file, n_est=100, subrow=.5, n_min=1):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename=f'{model_name}.log')

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    with open(feature_map_file) as f:
        feature_name = [x.strip() for x in f.readlines()]

    logging.info('Loading CV Ids')
    cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    p = np.zeros((X.shape[0], N_CLASS))
    p_tst = np.zeros((X_tst.shape[0], N_CLASS))
    for i, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
        logging.info(f'Training model #{i}')
        clf = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=n_min,
                                     max_features='auto', max_samples=subrow,
                                     random_state=SEED, n_jobs=-1)
        clf.fit(X[i_trn], y[i_trn])
        p[i_val, :] = clf.predict_proba(X[i_val])
        p_tst += clf.predict_proba(X_tst) / N_FOLD
        logging.info(f'CV #{i}: {accuracy_score(y[i_val], np.argmax(p[i_val], axis=1)) * 100:.4f}%')

    imp = pd.DataFrame({'feature': feature_name, 'importance': clf.feature_importances_})
    imp = imp.sort_values('importance').set_index('feature')
    imp.to_csv(feature_imp_file)

    logging.info(f'CV: {accuracy_score(y, np.argmax(p, axis=1)) * 100:.4f}%')
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--feature-imp-file', required=True, dest='feature_imp_file')
    parser.add_argument('--n-est', type=int, default=100, dest='n_est')
    parser.add_argument('--subrow', type=float, default=.5)
    parser.add_argument('--n-min', type=int, default=1, dest='n_min')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  feature_map_file=args.feature_map_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  feature_imp_file=args.feature_imp_file,
                  n_est=args.n_est,
                  subrow=args.subrow,
                  n_min=args.n_min)
    logging.info(f'finished ({(time.time() - start) / 60:.2f} min elasped)')
