import pandas as pd

# Define the bounding box coordinates (latitude, longitude)
# Replace these coordinates with the actual bounding box values
min_lat = 27.6877  # Minimum latitude
max_lat = 27.7111  # Maximum latitude
min_lon = 85.3188  # Minimum longitude
max_lon = 85.3404  # Maximum longitude

# Specify the file path
input_csv = 'ktm_selected_fields.csv'
output_csv = 'filtered_baneshwor_hattisar.csv'

# Read the CSV file
try:
    # Read GPS trace data from CSV file
    df = pd.read_csv(input_csv, delimiter=';', quotechar='"')

    # Filter data based on bounding box
    filtered_df = df[(df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) & 
                     (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)]

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_csv, index=False)

    print(f"Filtered GPS traces saved to {output_csv}")
except pd.errors.ParserError as e:
    print(f'ParserError: {e}')
except Exception as e:
    print(f'Error: {e}')
