import pandas as pd

input_csv_path = 'filtered_baneshwor_hattisar.csv'
output_csv_path = '301.csv'

# Ensure these match the actual column names in your CSV file
columns_to_keep = ['id', 'deviceid', 'servertime', 'latitude', 'longitude', 'speed']

chunks = []
chunk_size = 50000

try:
    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size, usecols=columns_to_keep, delimiter=',', quotechar='"', escapechar='\\'):
        chunks.append(chunk)

    df_selected = pd.concat(chunks, ignore_index=True)
    df_selected.to_csv(output_csv_path, index=False)
    print(f"Filtered data saved to {output_csv_path}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

