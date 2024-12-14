import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
data = pd.read_csv('master_9_sites.csv')

# List all columns in the dataset
print("Columns in the dataset:")
print(data.columns)

# Convert 'Date' to datetime for easier manipulation
data['Date'] = pd.to_datetime(data['Date'])

# Filter out rows where 'Si' is NaN
data = data[~data['Si'].isna()]

# Create the 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Create plots for each unique value in the 'Stream' column
for stream in data['Stream'].unique():
    # Filter data for the current stream
    stream_data = data[data['Stream'] == stream]

    # Check if there are valid Si values for the current stream
    if stream_data.empty:
        print(f"No valid Si values for stream: {stream}. Skipping.")
        continue

    # Determine the start and end dates
    start_date = stream_data['Date'].min()
    end_date = stream_data['Date'].max()

    # Filter stream data to only include rows within the start and end dates
    filtered_stream_data = stream_data[(stream_data['Date'] >= start_date) & (stream_data['Date'] <= end_date)]

    # Plot 'Si' against 'Date'
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_stream_data['Date'], filtered_stream_data['Si'],linestyle='-', label=stream)
    plt.title(f"Si Levels Over Time for {stream}", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Si Levels", fontsize=12)
    plt.xticks(filtered_stream_data['Date'].dt.to_period('Y').drop_duplicates().dt.to_timestamp(), rotation=45)  # Display years only
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG file in the 'plots' folder
    plt.savefig(f'plots/{stream}_Si_over_time.png')
    plt.close()

print("Plots saved in the 'plots' directory.")

# Also print the date ranges for each stream.
for stream in data['Stream'].unique():
    stream_data = data[data['Stream'] == stream]
    start_date = stream_data['Date'].min().strftime('%Y-%m-%d')
    end_date = stream_data['Date'].max().strftime('%Y-%m-%d')
    print(f"Stream: {stream}, Start Date: {start_date}, End Date: {end_date}")