import sys
sys.path.append("../")

import pandas as pd
import numpy as np

from prov_acquisition.prov_libraries.provenance_tracker import ProvenanceTracker

def main() -> None:

    df = pd.DataFrame({'key1': [0, 0, 1, 2, 0],
                       'key2': [0, np.nan, 0, 1, 0],
                       'A': [0, 1, 2, 3, 4],
                       'B': [0, 1, 2, 3, 4]
                       })
    right = pd.DataFrame({'key1': [0, np.nan, 1, 2 ],
                          'key2': [0, 4, 0, 0],
                          'A': [0, 1, 2, 3],
                          'D': [0, np.nan, 2, 3],
                          'C': [0, 1, 2, 3]})
    df2 = pd.DataFrame({'key1': [0, 5, 7, 10, 1],
                        'key2': [0, 4, 2, 1, 0],
                        'E': [1, 1, 2, 3, 9],
                        'F': [0, 1, 2, 3, 4]
                        })
    
    # Create provenance tracker
    tracker = ProvenanceTracker()

    df, right, df2 = tracker.subscribe([df, right, df2])

    # Instance generation
    df = df.append({'key2': 4}, ignore_index=True)

    # Join
    df = df.merge(right=right, on=['key1', 'key2'], how='left')

    # Imputation
    df = df.fillna(0)

    # Feature transformation of column D
    df['D'] = df['D'].apply(lambda x: x * 2)

     # Join
    df = df.merge(right=df2, on=['key1', 'key2'], how='left')

    # Feature transformation of column key2
    df['key2'] = df['key2'].apply(lambda x: x * 2)

    # Imputation 2
    df = df.fillna(10)
    
    # Space transformation 1
    tracker.dataframe_tracking = False
    c = 'D'
    dummies = pd.get_dummies(df[c])
    df_dummies = dummies.add_prefix(c + '_')
    df = df.join(df_dummies)
    tracker.dataframe_tracking = True
    df = df.drop([c], axis=1)

    # Space transformation 2
    tracker.dataframe_tracking = False
    c = 'E'
    dummies = pd.get_dummies(df[c])
    df_dummies = dummies.add_prefix(c + '_')
    df = df.join(df_dummies)
    tracker.dataframe_tracking = True
    df = df.drop([c], axis=1)

    df = df.drop(['B'], axis=1)
if __name__ == '__main__':
    main()
