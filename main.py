from prov_acquisition.prov_libraries.ProvenanceTracker import ProvenanceTracker
import pandas as pd

import numpy as np
import logging


def main():
    df = pd.DataFrame(np.random.randint(0, 10, size=(10, 4)), columns=list('ABCD')).astype(float)
    df['D'] = np.nan
    df[df < 7] = np.nan

    df2 = pd.DataFrame(np.random.randint(0, 10, size=(10, 4)), columns=list('AEFG')).astype(float)

    tracker = ProvenanceTracker(df)

    print(df)

    # Pipeline
    tracker.df_input.append({'A': 5}, ignore_index=True)
    tracker.df_input.applymap(func=lambda x: 2 if pd.isnull(x) else x // 3)
    #tracker.df_input.applymap(func=lambda x: np.nan if x % 2 == 0 else x)
    result = tracker.df_input.merge(right=df2, on='A', how='inner')
    print(result)
    result = tracker.df_input.drop('A', axis=1, )


    print(result)


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    logging.basicConfig(level=logging.INFO)

    main()
