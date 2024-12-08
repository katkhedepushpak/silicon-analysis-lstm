# preprocessing.py
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def load_and_preprocess_data(file_path, static_file_path, sequence_length):
    # Load data
    master_9_sites = pd.read_csv(file_path)
    master_9_sites.fillna(0, inplace=True)
    # print("Loaded master_9_sites data:")
    # print(master_9_sites.head())
    # print("Loaded master_9_sites data shape:", master_9_sites.shape)
    
    static = pd.read_csv(static_file_path)
    static.rename(columns={'Stream_Name': 'Stream'}, inplace=True)
    static.fillna(0, inplace=True)
    # print("Loaded static data:")
    # print(static.head())
    # print("Loaded static data shape:", static.shape)
    
    # Merge data
    merged_df = pd.merge(static, master_9_sites, on='Stream')
    # print("Merged data:")
    # print(merged_df.head())
    # print("Merged data shape:", merged_df.shape)
    
    # Reorder columns
    cols = [col for col in merged_df.columns if col != 'Si']
    new_cols = cols + ['Si']
    merged_df = merged_df[new_cols]
    # print("Reordered columns in merged data:")
    # print(merged_df.head())
    # print("Reordered columns in merged data shape:", merged_df.shape)
    
    # Map stream names to integers
    stream_mapping = {
        'CU11.6M': 1,
        'Kolyma': 2,
        'LMP': 3,
        'Lookout': 4,
        'Q1': 9, 
        'STREORK': 5,
        'Sagehen': 6,
        'Site.11564': 7,
        'WS2': 8
    }
    merged_df['Stream'] = merged_df['Stream'].map(stream_mapping)
    # print("Mapped stream names to integers:")
    # print(merged_df.head())
    # print("Mapped stream names to integers shape:", merged_df.shape)
    
    # Reorder columns
    date_column = merged_df['Date']
    merged_df.drop(labels=['Date'], axis=1, inplace=True)
    merged_df.insert(0, 'Date', date_column)

    
    # Split data by stream
    unique_streams = merged_df['Stream'].unique()
    stream_dataframes = {stream: merged_df[merged_df['Stream'] == stream] for stream in unique_streams}

    
    for stream in stream_dataframes:
        stream_dataframes[stream] = stream_dataframes[stream].sort_values(by='Date')
        sorted_stream =  stream_dataframes[stream] 
        nan_zero_count = (sorted_stream["Si"] == 0).sum()






    # Sort by date and extract year, month, day
    for stream, df in stream_dataframes.items():
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df.drop(columns=['Date'], inplace=True)
        stream_dataframes[stream] = df
    
    
    

    # Scale data
    scaler = MinMaxScaler()
    for stream, df in stream_dataframes.items():
        columns_to_scale = [col for col in df.columns if col not in ['Year', 'Month', 'Day', 'Stream', 'Si']]
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        stream_dataframes[stream] = df
    # print("Scaled data:")
    # for stream in stream_dataframes:
    #     print(f"Stream {stream}:")
    #     print(stream_dataframes[stream].head())
    #     print(f"Stream {stream} shape after scaling:", stream_dataframes[stream].shape)
    #     # in each stream print the count of non zero values for Si columns  
    #     print(f"Stream {stream} count of non zero values for Si columns:", stream_dataframes[stream][stream_dataframes[stream]['Si'] != 0].shape)
    
    # Split into train and test sets
    split_ratio = 0.8
    train_dataframes = {}
    test_dataframes = {}
    for stream, df in stream_dataframes.items():
        train_size = int(len(df) * split_ratio)
        train_dataframes[stream] = df[:train_size]
        test_dataframes[stream] = df[train_size:]
    # print("Split into train and test sets:")
    # for stream in train_dataframes:
    #     print(f"Train dataframe for stream {stream}: {train_dataframes[stream].shape}")
    #     print(f"Train dataframe for stream {stream} shape:", train_dataframes[stream].shape)
    #             # in each stream print the count of non zero values for Si columns  
    #     print(f"Train dataframe for stream {stream} count of non zero values for Si columns:", train_dataframes[stream][train_dataframes[stream]['Si'] != 0].shape)
    # for stream in test_dataframes:
    #     print(f"Test dataframe for stream {stream}: {test_dataframes[stream].shape}")
    #     print(f"Test dataframe for stream {stream} shape:", test_dataframes[stream].shape)
    #             # in each stream print the count of non zero values for Si columns
    #     print(f"Test dataframe for stream {stream} count of non zero values for Si columns:", test_dataframes[stream][test_dataframes[stream]['Si'] != 0].shape)
    
    # Create sequences
    def create_sequences(input_data: pd.DataFrame, target_column, sequence_length=3):
        sequences = []
        data_size = len(input_data)
        for i in tqdm(range(data_size - sequence_length)):
            sequence = input_data[i:i+sequence_length]
            label_position = i + sequence_length
            label = input_data.iloc[label_position][target_column]
            sequences.append((sequence, label))
        return sequences
    
    train_sequences = {stream: create_sequences(df, "Si", sequence_length) for stream, df in train_dataframes.items()}
    test_sequences = {stream: create_sequences(df, "Si", sequence_length) for stream, df in test_dataframes.items()}
    # print("Created sequences:")
    # for stream in train_sequences:
    #     print(f"Number of train sequences for stream {stream}: {len(train_sequences[stream])}")
    #     # in each stream print the count of non zero values for Si columns
    #     print(f"Train sequences for stream {stream} count of non zero values for Si columns:", len([seq for seq in train_sequences[stream] if seq[1] != 0.0]))
    # for stream in test_sequences:
    #     print(f"Number of test sequences for stream {stream}: {len(test_sequences[stream])}")
    #     # in each stream print the count of non zero values for Si columns 
    #     print(f"Test sequences for stream {stream} count of non zero values for Si columns:", len([seq for seq in test_sequences[stream] if seq[1] != 0.0]))
    
    # Verify the creation of sequences
    # for stream in train_sequences:
    #     print(f"Number of train sequences for stream '{stream}': {len(train_sequences[stream])}")
    #     lgth = len(train_sequences[stream])
    #     countr = 0
    #     for i in range(0, lgth):
    #         if (train_sequences[stream][i][1] != 0.0):
    #             countr = countr + 1
    #     print(f"{stream} : {countr}")

    # for stream in test_sequences:
    #     print(f"Number of test sequences for stream '{stream}': {len(test_sequences[stream])}")
    #     lgth = len(test_sequences[stream])
    #     countr = 0
    #     for i in range(0, lgth):
    #         if (test_sequences[stream][i][1] != 0.0 and not np.isnan(test_sequences[stream][i][1])):
    #             countr = countr + 1
    #     print(f"{stream} : {countr}")
    


    # # Filter sequences
    filtered_train_sequences = {stream: [(seq, label) for seq, label in sequences if not (np.isnan(label) or label == 0.0)] for stream, sequences in train_sequences.items()}
    filtered_test_sequences = {stream: [(seq, label) for seq, label in sequences if not (np.isnan(label) or label == 0.0)] for stream, sequences in test_sequences.items()}

    print(filtered_train_sequences[3][145])
    # # print Si values for 10 entries in each stream for test_sequences
    # for stream in filtered_test_sequences:
    #     print(f"Si values for 10 entries in stream '{stream}' for filtered_test_sequences:")
    #     for i in range(10):
    #         print(filtered_test_sequences[stream][i][1])
    # # print Si values for 10 entries in each stream for train_sequences
    # for stream in filtered_train_sequences:
    #     print(f"Si values for 10 entries in stream '{stream}' for filtered_train_sequences:")
    #     for i in range(10):
    #         print(filtered_train_sequences[stream][i][1])



    # # Merge sequences
    # merged_list_train = sum(filtered_train_sequences.values(), [])
    # merged_list_test = sum(filtered_test_sequences.values(), [])




    # Assign variables for filtered sequences for each stream in filtered_train_sequences
    train_sequences_CU11_6M = filtered_train_sequences[1]
    train_sequences_Kolyma = filtered_train_sequences[2]
    train_sequences_LMP = filtered_train_sequences[3]
    train_sequences_Lookout = filtered_train_sequences[4]
    train_sequences_Q1 = filtered_train_sequences[9]
    train_sequences_STREORK = filtered_train_sequences[5]
    train_sequences_Sagehen = filtered_train_sequences[6]
    train_sequences_Site11564 = filtered_train_sequences[7]
    train_sequences_WS2 = filtered_train_sequences[8]

    # Assign variables for filtered sequences for each stream in filtered_test_sequences
    test_sequences_CU11_6M = filtered_test_sequences[1]
    test_sequences_Kolyma = filtered_test_sequences[2]
    test_sequences_LMP = filtered_test_sequences[3]
    test_sequences_Lookout = filtered_test_sequences[4]
    test_sequences_Q1 = filtered_test_sequences[9]
    test_sequences_STREORK = filtered_test_sequences[5]
    test_sequences_Sagehen = filtered_test_sequences[6]
    test_sequences_Site11564 = filtered_test_sequences[7]
    test_sequences_WS2 = filtered_test_sequences[8]


    merged_list_train = train_sequences_CU11_6M + train_sequences_Kolyma + train_sequences_LMP + train_sequences_Lookout + \
                train_sequences_Q1 + train_sequences_STREORK + train_sequences_Sagehen + \
                train_sequences_Site11564 + train_sequences_WS2
    
    
    merged_list_test = test_sequences_CU11_6M + test_sequences_Kolyma + test_sequences_LMP + test_sequences_Lookout + \
                test_sequences_Q1 + test_sequences_STREORK + test_sequences_Sagehen + \
                test_sequences_Site11564 + test_sequences_WS2


    X_train_list = [torch.tensor(merged_list_train[i][0].values) for i in range(len(merged_list_train))]
    y_train_list = [torch.tensor(merged_list_train[i][1]) for i in range(len(merged_list_train))]

    # Ensure target tensors are 1-dimensional if they are scalars
    y_train_list = [y.unsqueeze(0) if y.dim() == 0 else y for y in y_train_list]

    # Stack the tensors to create the final X_train and y_train tensors
    X_train = torch.stack(X_train_list)
    y_train = torch.stack(y_train_list)
    X_train = X_train.float()
    y_train = y_train.float()

    # Convert DataFrames to tensors for the test data
    X_test_list = [torch.tensor(merged_list_test[i][0].values) for i in range(len(merged_list_test))]
    y_test_list = [torch.tensor(merged_list_test[i][1]) for i in range(len(merged_list_test))]

    # Ensure target tensors are 1-dimensional if they are scalars
    y_test_list = [y.unsqueeze(0) if y.dim() == 0 else y for y in y_test_list]

    # Stack the tensors to create the final X_test and y_test tensors
    X_test = torch.stack(X_test_list)
    y_test = torch.stack(y_test_list)

    # Ensure input tensors are of type torch.float32
    X_test = X_test.float()
    y_test = y_test.float()

    y_train1 = y_train
    y_test1 = y_test

    y_combined = np.concatenate([y_train, y_test])

    scaler_target = MinMaxScaler()
    scaler_target.fit(y_combined)

    y_train = scaler_target.transform(y_train)
    y_test = scaler_target.transform(y_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test, scaler_target, y_train1

