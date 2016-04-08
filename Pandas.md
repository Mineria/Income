# Pandas library useful functions

- Load CSV and create dataframe
  ```Python
  import pandas as pd
  import numpy as np

  # For .read_csv, always use header=0 when you know row 0 is the header row
  df = pd.read_csv('train.csv', header=0)
  ``
- Types for each column
  ```Python
  df.dtypes
  df.info()
  df.describe()
  ```

## Data Munging

- Referencing and filtering
  ```Python
  df['Age'][0:10]
  df[df['income'] == '>50K'][['age', 'sex', 'race']]
  df[df['income'].isnull()][['age', 'sex', 'race']]
  ```
- Get type of a column.
  ```Python
  type(df['Age'])
  ```
