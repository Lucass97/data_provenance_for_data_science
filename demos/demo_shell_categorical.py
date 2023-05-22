import sys
sys.path.append("../")

import pandas as pd
import numpy as np

from prov_acquisition.prov_libraries.provenance_tracker import ProvenanceTracker


def main() -> None:

    df = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2', 'K0'],
                       'key2': ['K0', np.nan, 'K0', 'K1', 'K0'],
                       'A': ['A0', 'A1', 'A2', 'A3', 'A4'],
                       'B': ['B0', 'B1', 'B2', 'B3', 'B4']
                       })
    right = pd.DataFrame({'key1': ['K0', np.nan, 'K1', 'K2', ],
                          'key2': ['K0', 'K4', 'K0', 'K0'],
                          'A': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', np.nan, 'D2', 'D3'],
                          'C': ['B0', 'B1', 'B2', 'B3']})
    df2 = pd.DataFrame({'key1': ['K0', 'imputato', 'K1', 'K1', 'K0'],
                        'key2': ['K0', 'K4', 'K2', 'K1', 'K0'],
                        'E': ['E1', 'E1', 'E2', 'E3', 'E4'],
                        'F': ['F0', 'F1', 'F2', 'F3', 'F4']
                        })

    # Create provenance tracker
    tracker = ProvenanceTracker()

    df, right, df2 = tracker.subscribe([df, right, df2])

    # Instance generation
    df = df.append({'key2': 'K4'}, ignore_index=True)

    # Join
    df = df.merge(right=right, on=['key1', 'key2'], how='left')

    # Imputation
    df = df.fillna('Imputation')

    # Feature transformation of column D
    df['D'] = df['D'].apply(lambda x: x * 2)

    # Feature transformation of column key2
    df['key2'] = df['key2'].apply(lambda x: x * 2)

    # Join
    df = df.merge(right=df2, on=['key1', 'key2'], how='left')

    # Feature transformation of column key2
    df['key2'] = df['key2'].apply(lambda x: x * 2)

    # Imputation 2

    df = df.fillna('Imputation')

    # Space transformation 1
    tracker.dataframe_tracking = False
    c = 'D'
    dummies = pd.get_dummies(df[c])
    df_dummies = dummies.add_prefix(c + '_')
    df = df.join(df_dummies)
    tracker.dataframe_tracking = True
    df = df.drop([c], axis=1)

    # Space transformation 2
    #tracker.dataframe_tracking = False
    c = 'E'
    dummies = pd.get_dummies(df[c])
    df_dummies = dummies.add_prefix(c + '_')
    df = df.join(df_dummies)
    #tracker.dataframe_tracking = True
    df = df.drop([c], axis=1)

    df = df.drop(['B'], axis=1)


if __name__ == '__main__':
    main()
