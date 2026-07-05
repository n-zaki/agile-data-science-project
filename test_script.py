print("Running automated test...")

import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3]})

assert len(df) == 3

print("Test passed successfully!")
