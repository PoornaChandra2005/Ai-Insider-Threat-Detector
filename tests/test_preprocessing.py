import pandas as pd
import numpy as np
from data_preprocess import feature_engineer, create_sequences

def test_feature_engineer():
    """Tests the feature engineering function."""
    data = {
        'date': ['2010-01-01 08:00:00', '2010-01-01 08:05:00', '2010-01-02 09:00:00'],
        'user': ['A', 'A', 'B'],
        'activity': ['Logon', 'Logoff', 'Logon']
    }
    df = pd.DataFrame(data)
    
    engineered_df = feature_engineer(df)
    
    assert 'time_delta' in engineered_df.columns
    assert 'hour' in engineered_df.columns
    assert 'day_of_week' in engineered_df.columns
    assert engineered_df['time_delta'].iloc[1] == 300.0 # 5 minutes * 60 seconds
    assert engineered_df['hour'].iloc[0] == 8
    assert engineered_df['day_of_week'].iloc[0] == 4 # Friday

def test_create_sequences():
    """Tests the sequence creation logic."""
    # 12 samples, 3 features + 1 label column
    data = np.random.rand(12, 4)
    data[:, -1] = np.random.randint(0, 2, 12) # last column is label
    seq_length = 5
    
    X, y = create_sequences(data, seq_length)
    
    # Expected number of sequences = len(data) - seq_length + 1 = 12 - 5 + 1 = 8
    assert X.shape == (8, seq_length, 3) # (num_sequences, seq_length, num_features)
    assert y.shape == (8,) # (num_sequences,)
    
    # Test if labeling works (if any event in sequence is 1, label is 1)
    test_data = np.zeros((6, 2)) # 5 samples, 1 feature + 1 label
    test_data[3, 1] = 1 # Mark the 4th event as an anomaly
    
    X_test, y_test = create_sequences(test_data, seq_length=4)
    
    # Sequences containing the anomaly at index 3:
    # seq 0: indices [0, 1, 2, 3] -> label should be 1
    # seq 1: indices [1, 2, 3, 4] -> label should be 1
    # seq 2: indices [2, 3, 4, 5] -> label should be 1
    assert y_test.tolist() == [1, 1, 1]