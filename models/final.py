import os
import pandas as pd
from datetime import datetime

def calculate_congestion_index(speed):
    if speed > 40:
        return "Free"
    elif 30 < speed <= 40:
        return "Basically Free"
    elif 20 < speed <= 30:
        return "Mild"
    elif 15 < speed <= 20:
        return "Moderate"
    elif speed < 15:
        return "Heavy"

def calculate_arrival_time(speed, length):
    return round((length / speed) * 60, 1)  # Convert hours to minutes and round to one decimal place

folder_path = 'road_segments_csv'

road_segment_lengths = {
    'maitighar_putalisadak': 1.15,
    'newbaneshwor_maitighar': 1.6,
    'newbaneshwor_oldbaneshwor': 1.53,
    'oldbaneshwor_setopool': 0.4,
    'putalisadak_hattisar': 0.53,
    'ratopool_hattisar': 1.43,
    'setopool_putalisadak': 1.3,
    'setopool_ratopool': 0.58
}

all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        road_segment = filename.split('.')[0] 
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        print(f"Processing file: {filename}")
        print("Columns in the file:", df.columns.tolist())

        if 'servertime' not in df.columns:
            print(f"'servertime' column not found in {filename}. Skipping this file.")
            continue

        df['road_segment'] = road_segment
        
        def is_peakhour(time):
            hour = datetime.strptime(time, '%Y/%m/%d %H:%M:%S').hour
            if (8 <= hour < 11) or (16 <= hour < 18):
                return 'peak'
            elif (6 <= hour < 8) or (11 <= hour < 16):
                return 'off_peak'
            else:
                return 'off_peak' 
        
        df['is_peakhour'] = df['servertime'].apply(is_peakhour)
        
        # Filter out rows with speed < 1 or speed > 60
        df = df[(df['speed'] >= 1) & (df['speed'] <= 60)]
        
        # Calculate congestion index
        df['congestion_index'] = df['speed'].apply(calculate_congestion_index)
        
        # Calculate arrival time
        length = road_segment_lengths[road_segment]
        df['arrival_time'] = df['speed'].apply(lambda x: calculate_arrival_time(x, length))
        
        # Convert servertime to YYYY-MM-DD format but keep the time
        df['servertime'] = df['servertime'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'))
        
        # Select necessary columns
        df = df[['servertime', 'road_segment', 'is_peakhour', 'speed', 'congestion_index', 'arrival_time']]
        df.rename(columns={'servertime': 'datetime'}, inplace=True)
        
        all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data)
    combined_df.to_csv('final_traffic_data.csv', index=False)
else:
    print("No data to combine. Please check the input CSV files.")
