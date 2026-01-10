from utils import load_df,save
import pandas as pd

pd.set_option("display.max_rows", None)

# load dataframes
final_df = load_df("output/final/data2.parquet")

print(final_df.columns)