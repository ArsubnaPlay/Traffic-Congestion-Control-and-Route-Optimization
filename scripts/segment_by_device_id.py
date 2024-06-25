import csv
import os

# Specify the input file path and output folder
input_file_path = 'csv/baneshwor_hattisar_updated.csv'
output_folder = 'csv_devices'

# Column names for the header row
header_row = ['id', 'deviceid', 'servertime', 'latitude', 'longitude', 'speed']

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the input CSV file
input_file = open(input_file_path, 'r')
input_reader = csv.reader(input_file)

# Dictionary to store the output files and writers for each device ID
output_files = {}

# Iterate over the rows in the input file
for row in input_reader:
    device_id = row[1]  # Assuming the device ID is in the second column

    # Create a new output file and writer for the current device ID if not already created
    if device_id not in output_files:
        output_file_path = os.path.join(output_folder, f'{device_id}.csv')
        output_file = open(output_file_path, 'w', newline='')
        output_writer = csv.writer(output_file)
        output_files[device_id] = (output_file, output_writer)

        # Write the header row to the output file
        output_writer.writerow(header_row)

    # Get the output file and writer for the current device ID
    output_file, output_writer = output_files[device_id]

    # Write the row to the corresponding output file
    output_writer.writerow(row)

# Close all output files
for output_file, _ in output_files.values():
    output_file.close()

# Close the input file
input_file.close()