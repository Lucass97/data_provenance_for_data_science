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
    parser = argparse.ArgumentParser(description="Real worlds pipelines - German Pipeline")
    parser.add_argument("--dataset", type=str, default="../../demos/real_world_pipelines/datasets/compas.csv",
                        help="Relative path to the dataset file")
    parser.add_argument("--frac", type=float, default=0.0, help="Sampling fraction")

    return parser.parse_args()

def run_pipeline(args) -> None:

    logger = CustomLogger('ProvenanceTracker')

    input_path = args.dataset

    df = pd.read_csv(input_path, header=0)

    if args.frac != 0.0:
        df = df.sample(frac=args.frac)
        logger.info(f'The dataframe was sampled ({args.frac * 100}%)')

    # Create provenance tracker
    tracker = ProvenanceTracker()

    # Subscribe dataframe
    df = tracker.subscribe(df)


    logger.info(f' OPERATION B0 - Selection of 9 relevant columns')

    columns = ['age', 'c_charge_degree', 'race', 'sex', 'priors_count', 'days_b_screening_arrest', 'two_year_recid', 'c_jail_in', 'c_jail_out']
    df = df.drop(df.columns.difference(columns), axis=1)


    logger.info(f' OPERATION B1 - Remove missing values')

    df = df.dropna()

    logger.info(f' OPERATION B2 - Make race feature binary')

    df['race'] = [0 if r != 'Caucasian' else 1 for r in df['race']]


    logger.info(f' OPERATION C3 - Rename two_year_recid column to label and value trasformation of the label column')
    
    tracker.dataframe_tracking = False
    df = df.rename({'two_year_recid': 'label'}, axis=1)

    tracker.dataframe_tracking = True
	# Reverse label for consistency with function defs: 1 means no recid (good), 0 means recid (bad)
    df['label'] = [0 if l == 1 else 1 for l in df['label']]


    logger.info(f' OPERATION B4 - Create jailtime column and convert it to days')

    df['jailtime'] = (pd.to_datetime(df.c_jail_out) - pd.to_datetime(df.c_jail_in)).dt.days


    logger.info(f' OPERATION B5 - Drop c_jail_in and c_jail_out features')

    df = df.drop(['c_jail_in', 'c_jail_out'], axis=1)
	

    logger.info(f' OPERATION B6 - Value transformation of column c_charge_degree')

	# M: misconduct, F: felony
    df['c_charge_degree'] = [0 if s == 'M' else 1 for s in df['c_charge_degree']]


if __name__ == '__main__':
    run_pipeline(get_args())