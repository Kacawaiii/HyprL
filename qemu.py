import pandas as pd

bin_df = pd.read_csv("data/supersearch_binary_aapl_1y_cli.csv")
amp_df = pd.read_csv("data/supersearch_amplitude_aapl_1y_cli.csv")

print("Binary survivors:", len(bin_df))
print("Amplitude survivors:", len(amp_df))

if not bin_df.empty:
    print("\nBinary top 5:")
    print(bin_df[["profit_factor", "expectancy", "num_trades"]].head())

if not amp_df.empty:
    print("\nAmplitude top 5:")
    print(amp_df[["profit_factor", "expectancy", "num_trades"]].head())
