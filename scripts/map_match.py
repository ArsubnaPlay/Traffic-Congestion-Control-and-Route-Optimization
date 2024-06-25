import csv
import folium
import webbrowser
import os
import pandas as pd

all_coordinates = []

def extract_coordinates(file_path, lat_column, lon_column, encoding='utf-8'):
    df = pd.read_csv(file_path, encoding=encoding,nrows=10000)
    # Drop rows with NaN values in the latitude and longitude columns
    df = df.dropna(subset=[lat_column, lon_column])
    
    valid_coordinates = []
    for index, row in df.iterrows():
        try:
            lat = float(row[lat_column])
            lon = float(row[lon_column])
            valid_coordinates.append([lat, lon])
        except ValueError as e:
            print(f"Skipping invalid row at index {index} in {file_path}: {row.to_dict()} - {e}")

    all_coordinates.extend(valid_coordinates)
    return valid_coordinates

# List of file paths
file_paths = [
    'filtered_baneshwor_hattisar.csv',
] 
lat_column = 'latitude'
lon_column = 'longitude'

# Extract coordinates from all files
for file_path in file_paths:
    extract_coordinates(file_path, lat_column, lon_column)

# Initialize the map
mymap = folium.Map(location=[27.6991, 85.3371], tiles="Cartodb Positron", zoom_start=14)

print(all_coordinates)
# Add each coordinate as a marker
for coord in all_coordinates:
    folium.Marker(location=coord).add_to(mymap)

# Save the map to an HTML file
mymap.save('map_matched_hmm.html')

# Open the map in the default web browser
webbrowser.open('file://' + os.path.realpath('map_matched_hmm.html'))