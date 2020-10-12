import argparse
from const import N_CLASS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-models', required=True, nargs='+', dest='base_models')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    with open(args.feature_map_file, 'w') as f:
        for i, col in enumerate(args.base_models):
            for i_class in range(N_CLASS):
                f.write(f'{col}_{i_class}\n')
