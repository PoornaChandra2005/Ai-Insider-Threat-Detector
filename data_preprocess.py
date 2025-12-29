import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import networkx as nx

# Constants
SEQ_LENGTH = 7
SEED = 42

def set_seed(seed):
    np.random.seed(seed)

def load_all_data(data_path):
    print("Loading all data sources...")
    try:
        logon_df = pd.read_csv(os.path.join(data_path, 'logon.csv'))
        device_df = pd.read_csv(os.path.join(data_path, 'device.csv'))
        file_df = pd.read_csv(os.path.join(data_path, 'file.csv'))
        http_df = pd.read_csv(os.path.join(data_path, 'http.csv'))
        email_df = pd.read_csv(os.path.join(data_path, 'email.csv'))
        psychometric_df = pd.read_csv(os.path.join(data_path, 'psychometric.csv'))
        for df in [logon_df, device_df, file_df, http_df, email_df]:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return logon_df, device_df, file_df, http_df, email_df, psychometric_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all CSV files are in '{data_path}'.")
        exit()

def engineer_daily_features(logon_df, device_df, file_df, http_df, email_df):
    print("Engineering daily summary features...")
    all_dates = pd.to_datetime(logon_df['date'].dt.date.unique())
    all_users = logon_df['user'].unique()
    user_date_grid = pd.MultiIndex.from_product([all_users, all_dates], names=['user', 'date'])
    daily_summary = pd.DataFrame(index=user_date_grid).reset_index()
    daily_summary['date'] = pd.to_datetime(daily_summary['date'])
    for df in [logon_df, device_df, file_df, http_df, email_df]:
        df['day'] = df['date'].dt.date
    def aggregate_and_merge(base_df, feature_df, agg_dict):
        agg_df = feature_df.groupby(['user', 'day']).agg(**agg_dict).reset_index()
        agg_df['day'] = pd.to_datetime(agg_df['day'])
        return pd.merge(base_df, agg_df, how='left', left_on=['user', 'date'], right_on=['user', 'day']).drop(columns=['day'])
    daily_summary = aggregate_and_merge(daily_summary, logon_df, agg_dict={'logon_count': ('id', 'count')})
    logon_df['hour'] = logon_df['date'].dt.hour
    logon_df['is_after_hours'] = ((logon_df['hour'] < 8) | (logon_df['hour'] > 18)).astype(int)
    daily_summary = aggregate_and_merge(daily_summary, logon_df, agg_dict={'after_hours_logon_count': ('is_after_hours', 'sum')})
    daily_summary = aggregate_and_merge(daily_summary, device_df[device_df['activity'] == 'Connect'], agg_dict={'usb_connect_count': ('id', 'count')})
    daily_summary = aggregate_and_merge(daily_summary, file_df, agg_dict={'file_copy_count': ('id', 'count')})
    daily_summary = aggregate_and_merge(daily_summary, http_df, agg_dict={'http_visit_count': ('id', 'count')})
    daily_summary = aggregate_and_merge(daily_summary, email_df, agg_dict={'email_sent_count': ('id', 'count')})
    feature_cols = [col for col in daily_summary.columns if col not in ['user', 'date']]
    daily_summary[feature_cols] = daily_summary[feature_cols].fillna(0)
    return daily_summary

def engineer_graph_features(daily_summary, email_df):
    print("Engineering graph-based email features...")
    email_df['to_list'] = email_df['to'].str.split(';').fillna("").apply(list)
    train_cutoff_date = email_df['date'].quantile(0.8)
    historical_emails = email_df[email_df['date'] <= train_cutoff_date]
    known_contacts = historical_emails.groupby('user')['to_list'].apply(lambda lists: set(r for sublist in lists for r in sublist))
    daily_new_contacts = []
    email_df_sorted = email_df.sort_values('date')
    for (user, day), group in email_df_sorted.groupby(['user', 'day']):
        daily_recipients = set(r for sublist in group['to_list'] for r in sublist)
        new_contacts = daily_recipients - known_contacts.get(user, set())
        daily_new_contacts.append({'user': user, 'day': day, 'new_contacts_count': len(new_contacts)})
    df_new_contacts = pd.DataFrame(daily_new_contacts)
    df_new_contacts['day'] = pd.to_datetime(df_new_contacts['day'])
    daily_summary = pd.merge(daily_summary, df_new_contacts, how='left', left_on=['user', 'date'], right_on=['user', 'day']).drop(columns=['day'])
    daily_summary['new_contacts_count'] = daily_summary['new_contacts_count'].fillna(0)
    return daily_summary

def create_advanced_labels(daily_summary):
    print("Creating refined, high-confidence anomaly labels...")
    feature_cols = [col for col in daily_summary.columns if col not in ['user', 'date', 'anomaly']]
    user_baselines = daily_summary.groupby('user')[feature_cols].agg(['mean', 'std']).reset_index()
    user_baselines.columns = ['_'.join(col).strip() for col in user_baselines.columns.values]
    user_baselines = user_baselines.rename(columns={'user_': 'user'})
    daily_summary = pd.merge(daily_summary, user_baselines, on='user', how='left')
    rule1 = (daily_summary['file_copy_count'] > daily_summary['file_copy_count_mean'] + 2 * daily_summary['file_copy_count_std']) & (daily_summary['after_hours_logon_count'] > 0)
    rule2 = (daily_summary['new_contacts_count'] > daily_summary['new_contacts_count_mean'] + 2 * daily_summary['new_contacts_count_std']) & (daily_summary['new_contacts_count'] > 5)
    rule3 = (daily_summary['usb_connect_count'] > daily_summary['usb_connect_count_mean'] + 2 * daily_summary['usb_connect_count_std']) & (daily_summary['logon_count'] > daily_summary['logon_count_mean'] + 2 * daily_summary['logon_count_std'])
    daily_summary['anomaly'] = (rule1 | rule2 | rule3).astype(int)
    num_anomalies = daily_summary['anomaly'].sum()
    print(f"Labeled {num_anomalies} user-days as anomalous ({num_anomalies / len(daily_summary):.2%}).")
    daily_summary = daily_summary.drop(columns=[c for c in daily_summary.columns if '_mean' in c or '_std' in c])
    return daily_summary

def create_sequences(user_data, features, seq_length):
    sequences, seq_labels, seq_users, seq_dates = [], [], [], []
    for i in range(len(user_data) - seq_length + 1):
        sequence_window = user_data.iloc[i:i + seq_length]
        sequences.append(sequence_window[features].values)
        seq_labels.append(int(np.any(sequence_window['anomaly'].values)))
        seq_users.append(sequence_window['user'].iloc[-1])
        seq_dates.append(sequence_window['date'].iloc[-1])
    return sequences, seq_labels, seq_users, seq_dates

def main(data_path, output_path):
    set_seed(SEED)
    os.makedirs(output_path, exist_ok=True)
    logon, device, file, http, email, psychometric = load_all_data(data_path)
    daily_summary = engineer_daily_features(logon, device, file, http, email)
    daily_summary = engineer_graph_features(daily_summary, email)
    daily_summary = create_advanced_labels(daily_summary)
    if 'user_id' in psychometric.columns: psychometric.rename(columns={'user_id': 'user'}, inplace=True)
    daily_summary = pd.merge(daily_summary, psychometric[['user', 'O', 'C', 'E', 'A', 'N']], on='user', how='left')
    daily_summary[['O', 'C', 'E', 'A', 'N']] = daily_summary[['O', 'C', 'E', 'A', 'N']].fillna(0.5)

    print("Splitting data by user...")
    users_with_anomalies = daily_summary[daily_summary['anomaly'] == 1]['user'].unique()
    normal_users = daily_summary[~daily_summary['user'].isin(users_with_anomalies)]['user'].unique()
    test_anomaly_users, train_anomaly_users = train_test_split(users_with_anomalies, test_size=0.7, random_state=SEED)
    if len(test_anomaly_users) > 0 and len(normal_users) > 0:
        num_normal_test = min(len(test_anomaly_users), len(normal_users))
        num_normal_train = min(len(train_anomaly_users), len(normal_users) - num_normal_test)
        if num_normal_test == len(normal_users):
            test_normal_users, train_normal_users = normal_users, np.array([])
        else:
            test_normal_users, remaining_normal_users = train_test_split(normal_users, train_size=num_normal_test, random_state=SEED)
            if num_normal_train > 0 and len(remaining_normal_users) > 0:
                train_size_final = min(num_normal_train, len(remaining_normal_users))
                train_normal_users, _ = train_test_split(remaining_normal_users, train_size=train_size_final, random_state=SEED)
            else:
                train_normal_users = np.array([])
    else:
        test_normal_users, train_normal_users = train_test_split(normal_users, test_size=0.3, random_state=SEED)
    train_users = np.concatenate([train_anomaly_users, train_normal_users])
    test_users = np.concatenate([test_anomaly_users, test_normal_users])
    train_set = daily_summary[daily_summary['user'].isin(train_users)].sort_values(by=['user', 'date'])
    test_set = daily_summary[daily_summary['user'].isin(test_users)].sort_values(by=['user', 'date'])
    print(f"Training on {len(train_users)} users. Testing on {len(test_users)} users.")

    features = [col for col in daily_summary.columns if col not in ['user', 'date', 'anomaly']]
    scaler = MinMaxScaler()
    train_set_scaled, test_set_scaled = train_set.copy(), test_set.copy()
    train_set_scaled[features] = scaler.fit_transform(train_set[features])
    test_set_scaled[features] = scaler.transform(test_set[features])
    joblib.dump(scaler, os.path.join(output_path, 'daily_summary_scaler.joblib'))

    print("\nSaving data for XGBoost...")
    train_set_scaled.to_csv(os.path.join(output_path, 'xgb_train.csv'), index=False)
    test_set_scaled.to_csv(os.path.join(output_path, 'xgb_test.csv'), index=False)
    
    print("Creating and saving sequences for LSTM...")
    all_train_sequences, all_test_sequences, all_test_labels = [], [], []
    seq_test_users, seq_test_dates = [], []
    train_users_normal = train_set[train_set['anomaly'] == 0]['user'].unique()
    for user in train_users_normal:
        user_data = train_set_scaled[train_set_scaled['user'] == user]
        if len(user_data) >= SEQ_LENGTH:
            sequences, _, _, _ = create_sequences(user_data, features, SEQ_LENGTH)
            all_train_sequences.extend(sequences)
    for user in test_users:
        user_data = test_set_scaled[test_set_scaled['user'] == user]
        if len(user_data) >= SEQ_LENGTH:
            sequences, labels, users, dates = create_sequences(user_data, features, SEQ_LENGTH)
            all_test_sequences.extend(sequences)
            all_test_labels.extend(labels)
            seq_test_users.extend(users)
            seq_test_dates.extend(dates)
    X_train, X_test, y_test = np.array(all_train_sequences), np.array(all_test_sequences), np.array(all_test_labels)
    ids_test = pd.DataFrame({'user': seq_test_users, 'date': seq_test_dates})
    np.savez_compressed(os.path.join(output_path, 'lstm_train.npz'), X=X_train)
    np.savez_compressed(os.path.join(output_path, 'lstm_test.npz'), X=X_test, y=y_test)
    ids_test.to_csv(os.path.join(output_path, 'lstm_test_ids.csv'), index=False)
    
    print("\nPreprocessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced preprocessing with graph features.")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the raw CERT dataset directory.")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the processed data files.")
    parser.add_argument('--seed', type=int, default=SEED, help="Random seed for reproducibility.")
    args = parser.parse_args()
    main(args.data_path, args.output_path)