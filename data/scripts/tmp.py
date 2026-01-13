from utils import load_df,save
import pandas as pd

pd.set_option("display.max_rows", None)

# load dataframes
final_df = load_df("data/processed/rdf_results_final.parquet")

print(final_df.shape)