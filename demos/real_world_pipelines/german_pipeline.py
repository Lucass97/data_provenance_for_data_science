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
    parser.add_argument("--dataset", type=str, default="../../demos/real_world_pipelines/datasets/german.csv",
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


    logger.info(f' OPERATION A0 - Value transformation of 13 distinc columns')

    df = df.replace({'checking': {'A11': 'check_low', 'A12': 'check_mid', 'A13': 'check_high',
                                  'A14': 'check_none'},
                     'credit_history': {'A30': 'debt_none', 'A31': 'debt_noneBank',
                                        'A32': 'debt_onSchedule', 'A33': 'debt_delay',
                                        'A34': 'debt_critical'},
                     'purpose': {'A40': 'pur_newCar', 'A41': 'pur_usedCar',
                                 'A42': 'pur_furniture', 'A43': 'pur_tv',
                                 'A44': 'pur_appliance', 'A45': 'pur_repairs',
                                 'A46': 'pur_education', 'A47': 'pur_vacation',
                                 'A48': 'pur_retraining', 'A49': 'pur_business',
                                 'A410': 'pur_other'},
                     'savings': {'A61': 'sav_small', 'A62': 'sav_medium', 'A63': 'sav_large',
                                 'A64': 'sav_xlarge', 'A65': 'sav_none'},
                     'employment': {'A71': 'emp_unemployed', 'A72': 'emp_lessOne',
                                    'A73': 'emp_lessFour', 'A74': 'emp_lessSeven',
                                    'A75': 'emp_moreSeven'},
                     'other_debtors': {'A101': 'debtor_none', 'A102': 'debtor_coApp',
                                       'A103': 'debtor_guarantor'},
                     'property': {'A121': 'prop_realEstate', 'A122': 'prop_agreement',
                                  'A123': 'prop_car', 'A124': 'prop_none'},
                     'other_inst': {'A141': 'oi_bank', 'A142': 'oi_stores', 'A143': 'oi_none'},
                     'housing': {'A151': 'hous_rent', 'A152': 'hous_own', 'A153': 'hous_free'},
                     'job': {'A171': 'job_unskilledNR', 'A172': 'job_unskilledR',
                             'A173': 'job_skilled', 'A174': 'job_highSkill'},
                     'phone': {'A191': 0, 'A192': 1},
                     'foreigner': {'A201': 1, 'A202': 0},
                     'label': {2: 0}})
    
    

    logger.info(f' OPERATION A1 - Generation of two new column from the column personal_status')

    tracker.dataframe_tracking = False
    # Translate status values
    df['status'] = np.where(df.personal_status == 'A91', 'divorced',
                            np.where(df.personal_status == 'A92', 'divorced',
                                     np.where(df.personal_status == 'A93', 'single',
                                              np.where(df.personal_status == 'A95', 'single',
                                                       'married'))))
    tracker.dataframe_tracking = True
    # Translate gender values
    df['gender'] = np.where(df.personal_status == 'A92', 0, np.where(df.personal_status == 'A95', 0, 1))
    
    
    logger.info(f' OPERATION A2 - Drop personal_status column')

    df = df.drop(['personal_status'], axis=1)


    logger.info(f' OPERATION A3 - One-hot encode of 11 categorical columns')

    
    tracker.dataframe_tracking = False

    columns = ['checking', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property',
           'other_inst', 'housing', 'job', 'status']

    for i, col in enumerate(columns):
        
        dummies = pd.get_dummies(df[col])
        df_dummies = dummies.add_prefix(col + '_')
        df = df.join(df_dummies)
        
        # Check last iteration:
        if i == len(columns) - 1:
            tracker.dataframe_tracking = True
        
        df = df.drop([col], axis=1)
    

if __name__ == '__main__':
    run_pipeline(get_args())
