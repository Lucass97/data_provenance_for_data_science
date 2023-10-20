from codecs import ignore_errors
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

    if args.frac > 0.0 and args.frac < 1.0:
        df = df.sample(frac=args.frac)
        logger.info(f'The dataframe was sampled ({args.frac * 100}%)')
    elif args.frac > 1.0:
        df = pd.concat([df] * int(args.frac), ignore_index=True)
        logger.info(f'The dataframe has been enlarged by ({int(args.frac)} times')
    
    df['index'] = df.index

    # Create provenance tracker
    tracker = ProvenanceTracker(save_on_neo4j=False)

    logger.info(f'{df.columns}')
    logger.info(f'{len(df)}')
    #logger.info(f'{df.memory_usage()}')
    
    # Subscribe dataframe
    df = tracker.subscribe(df)

    logger.info(f' OPERATION DR - Drop fnlwgt column')

    df = df.drop(['fnlwgt'], axis=1)

    logger.info(f' OPERATION FT - Assign sex and label binary values 0 and 1')

    tracker.dataframe_tracking = False
    df['sex'] = df['sex'].apply(str.strip)
    tracker.dataframe_tracking = True
    df = df.replace({'sex': {'Male': 1, 'Female': 0}})

    logger.info(f' OPERATION I - Imputation')

    df = df.fillna("?")

    logger.info(f' OPERATION ST- Space transformation')

    df["capital"] = df["capital-gain"] - df["capital-loss"]
    
    logger.info(f' OPERATION IG - Instance Generation')

    df = df.append({'age': 77}, ignore_index=True)
    
    logger.info(f' OPERATION VT - Value Transformation')

    df["hours-per-week"] = df["hours-per-week"] * 60
    
    logger.info(f' OPERATION JO - Join two records')

    df = df.merge(right=df, on=['index'], how='left')
    
    logger.info(f' OPERATION AP - Append onto a table')

    df.append(df, ignore_index=True)



if __name__ == '__main__':
    run_pipeline(get_args())