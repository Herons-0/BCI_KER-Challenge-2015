import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import os


# Helper function to extract 1-second P300 windows starting 300ms after each feedback event
def extract_p300_windows(df, event_column='FeedBackEvent', signal_columns=None, pre_offset=0.3, duration=1.0):
   
    if signal_columns is None:
        # Exclude non-EEG columns like 'Time' and 'FeedBackEvent'
        signal_columns = [col for col in df.columns if col not in ['Time', event_column]]
    
    # Estimate sampling rate from time column
    time_diffs = df['Time'].diff().dropna()
    sampling_interval = time_diffs.median()
    sampling_rate = int(round(1.0 / sampling_interval))

    # Convert durations to number of samples
    start_offset_samples = int(pre_offset * sampling_rate)
    window_size_samples = int(duration * sampling_rate)
    
    # Extract segments
    segments = []
    event_times = []

    feedback_indices = df.index[df[event_column] != 0].tolist()
    
    for idx in feedback_indices:
        start_idx = idx + start_offset_samples
        end_idx = start_idx + window_size_samples
        if end_idx < len(df):
            window = df.iloc[start_idx:end_idx][signal_columns].to_numpy()
            segments.append(window)
            event_times.append(df.iloc[idx]['Time'])

    return segments, event_times

final_segments =[]
final_feedback_times = []

# Load all the CSV files from the folder given on the command line:
#   python model.py path/to/train
import sys

if len(sys.argv) != 2:
    sys.exit("usage: python model.py <folder with the training CSVs>")
folder_path = sys.argv[1]

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file in csv_files:
    filepath = os.path.join(folder_path, file)
    try:
        print("Reading:", filepath)
        df = pd.read_csv(filepath)
        print("Shape:", df.shape)
        eeg_segments, feedback_times = extract_p300_windows(df)
        eeg_segments = np.array(eeg_segments)
        eeg_segments = eeg_segments.reshape(-1, 11400)  # reshaped to (num_segments, 11400) for MLP input
        print("Extracted segments:", len(eeg_segments), "with shape:", eeg_segments.shape)
        final_segments.append(eeg_segments)
        final_feedback_times.append(feedback_times)

    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
print(final_feedback_times)
# final_feedback_times = np.array(final_feedback_times)
final_segments = np.array(final_segments)
print("Total segments extracted:", len(final_segments))
print(final_feedback_times.shape)
print(final_segments.shape)
arr = np.hstack((final_segments,final_feedback_times))
indices = np.arange(arr.shape[0]).reshape(-1, 1)  # Shape: (n, 1)
letter_Number = np.tile([1,2,3,4,5],12)
Train_segments = np.hstack((indices, arr))  # Combine indices with Train_segments
Train_segments = np.hstack((Train_segments, letter_Number.reshape(-1, 1)))  


# Show shape of first segment and number of segments
segment_shape = Train_segments[0].shape if eeg_segments else (0, 0)
print(len(eeg_segments), segment_shape)


# print(eeg_segments)
# print(feedback_times)

hlowrld = MLPClassifier(
    hidden_layer_sizes=(11404,1000,1000,2), # 56 EEG + 1 EOG + 1 Time + 1 Session + 1 Word Number + 1 Character Number Number times 200
    activation='relu',
    solver='adam',
    alpha = 3e-5,
    batch_size=200,
    # learning_rate='adaptive',
    learning_rate_init=0.01,
    max_iter=200,
    shuffle=True,
    random_state=42,
    tol=1e-4,
    verbose=True,
    warm_start=True,
    early_stopping=True,
    validation_fraction=0.2,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    n_iter_no_change=10,
)

# hlowrld.fit()