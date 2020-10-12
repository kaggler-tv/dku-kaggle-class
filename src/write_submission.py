#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd

from const import ID_COL, TARGET_COL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-file', required=True, dest='predict_file')
    parser.add_argument('--sample-file', required=True, dest='sample_file')
    parser.add_argument('--output-file', required=True, dest='output_file')
    args = parser.parse_args()

    p = np.loadtxt(args.predict_file, delimiter=',')
    sub = pd.read_csv(args.sample_file, index_col=ID_COL)
    sub[TARGET_COL] = np.argmax(p, axis=1)
    sub.to_csv(args.output_file)
