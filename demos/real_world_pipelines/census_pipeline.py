import sys

sys.path.append("../../")

import argparse
import pandas as pd
import numpy as np
from misc.logger import CustomLogger

from prov_acquisition.prov_libraries.tracker import ProvenanceTracker

def get_args() -> argparse.Namespace:
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Real worlds pipelines - Census Pipeline")
    parser.add_argument("--dataset", type=str, default="../../demos/real_world_pipelines/datasets/census.csv",
                        help="Relative path to the dataset file")
    parser.add_argument("--frac", type=float, default=0.0, help="Sampling fraction [0.0 - 1.0]")

    return parser.parse_args()

def run_pipeline(args) -> None:
     
    logger = CustomLogger('ProvenanceTracker')

    input_path = args.dataset

    df = pd.read_csv(input_path)

    # Assign names to columns
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']

    df.columns = names

    if args.frac != 0.0:
        df = df.sample(frac=args.frac)
        logger.info(f'The dataframe was sampled ({args.frac * 100}%)')

    # Create provenance tracker
    tracker = ProvenanceTracker()
    
    # Subscribe dataframe
    df = tracker.subscribe(df)

    
    logger.info(f' OPERATION C0 - Remove whitespace from 9 columns')

    columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country',
           'label']

    df[columns] = df[columns].applymap(str.strip)


    logger.info(f' OPERATION C1 - Replace ? character for NaN value')
    
    df = df.replace('?', np.nan)
    

    logger.info(f' OPERATION C2 - One-hot encode 7 categorical features')

    tracker.dataframe_tracking = False
    
    columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    columns = ['education']

    for i, col in enumerate(columns):
            
        dummies = pd.get_dummies(df[col])
        df_dummies = dummies.add_prefix(col + '_')
        df = df.join(df_dummies)
        
        # Check last iteration:
        if i == len(columns) - 1:
            tracker.dataframe_tracking = True
        
        df = df.drop([col], axis=1)


    logger.info(f' OPERATION C3 - Assign sex and label binary values 0 and 1')

    df = df.replace({'sex': {'Male': 1, 'Female': 0}, 'label': {'<=50K': 0, '>50K': 1}})


    logger.info(f' OPERATION C4 - Drop fnlwgt column')

    df = df.drop(['fnlwgt'], axis=1)


if __name__ == '__main__':
    run_pipeline(get_args())