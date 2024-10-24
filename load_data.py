# import tensorflow as tf
import pandas as pd
import os,tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


#First we group by vehicle_id and we will forward fill the last known value.
#Then if the entire column is NaN, we will fill it with the median of the column. 
#If there are still any NaNs we will fill them with 0.
def fill_missing_values(df):
    df = df.groupby('vehicle_id').apply(lambda x: x.ffill(axis=0)) #Forward fill last known value, but only for the same vehicle
    df = df.droplevel('vehicle_id') #Remove multi-index, as we don't want to group by vehicle_id anymore    
    df = df.fillna(df.median()) #Fill with median rather than mean to avoid outliers
    df = df.fillna(0) #Last resort fill with 0
    return df


def clean_readouts(readouts):
    #Clean the data
    print(f'Number of missing values before cleaning: {readouts.isnull().sum().sum()}')
    readouts = fill_missing_values(readouts)
    print(f'Number of missing values after cleaning: {readouts.isnull().sum().sum()}')
    #readoutsValidation = fill_missing_values(readoutsValidation)
    return readouts


def get_class_label(row):
    #classes denoted by 0, 1, 2, 3, 4 where they are related to readouts within a time window of: (more than 48), (48 to 24), (24 to 12), (12 to 6), and (6 to 0) time_step before the failure, respectively
    if row['time_to_potential_event'] > 48:
        return 0 #No failure within 48 time steps
    elif row['time_to_potential_event'] > 24 and row['in_study_repair'] == 1:
        return 1 #Failure within 48 to 24 time steps
    elif row['time_to_potential_event'] > 12 and row['in_study_repair'] == 1:
        return 2 #Failure within 24 to 12 time steps
    elif row['time_to_potential_event'] > 6 and row['in_study_repair'] == 1:
        return 3 #Failure within 12 to 6 time steps
    elif row['time_to_potential_event'] > 0 and row['in_study_repair'] == 1:
        return 4 #Failure within 6 to 0 time steps
    else:
        return 5 #No failure reported, but within 48 time steps from the end of the study, don't know if it will fail or not
    
def add_class_labels(tte, readouts):
    # Join the readouts and the time to event data
    df = pd.merge(readouts, tte, on = 'vehicle_id', how='left').copy()
    #Calculate the time to a failure event
    df['time_to_potential_event'] = df['length_of_study_time_step'] - df['time_step']
    df['class_label'] = df.apply(get_class_label, axis=1)
    return df


def load_and_prepare_dataset(dataset_dir):

    fnames = ['test_labels.csv',
            'test_operational_readouts.csv',
            'test_specifications.csv',
            'train_operational_readouts.csv',
            'train_specifications.csv',
            'train_tte.csv',
            'validation_labels.csv',
            'validation_operational_readouts.csv',
            'validation_specifications.csv']

    train_readouts = pd.read_csv(os.path.join(dataset_dir, 'train_operational_readouts.csv'))
    train_spec = pd.read_csv(os.path.join(dataset_dir, 'train_specifications.csv'))
    train_tte = pd.read_csv(os.path.join(dataset_dir, 'train_tte.csv'))

    val_readouts = pd.read_csv(os.path.join(dataset_dir, 'validation_operational_readouts.csv'))
    val_spec = pd.read_csv(os.path.join(dataset_dir, 'validation_specifications.csv'))
    val_tte = pd.read_csv(os.path.join(dataset_dir, 'validation_labels.csv'))

    test_readouts = pd.read_csv(os.path.join(dataset_dir, 'test_operational_readouts.csv'))
    test_spec = pd.read_csv(os.path.join(dataset_dir, 'test_specifications.csv'))
    test_tte = pd.read_csv(os.path.join(dataset_dir, 'test_labels.csv'))

    train_readouts = fill_missing_values(train_readouts)
    val_readouts = fill_missing_values(val_readouts)
    test_readouts = fill_missing_values(test_readouts)

    df_train = add_class_labels(train_tte, train_readouts)

    df_train = df_train.drop(['length_of_study_time_step','in_study_repair'],axis=1)

    return df_train

def stratified_train_test_split(df_train):
    train_X = []
    train_Y = []

    # More efficient processing using groupby
    grouped = df_train.groupby('vehicle_id')
    
    # Find the maximum group length
    max_length = grouped.size().max()
    print(f"Maximum sequence length: {max_length}")
    
    # You can also see the distribution of sequence lengths
    group_sizes = grouped.size()
    print("\nSequence length statistics:")
    print(group_sizes.describe())
    
    for vehicle_id, group in tqdm.tqdm(grouped):
        # Extract features in one operation per group
        tr_x = group.drop(['time_to_potential_event', 'class_label', 'vehicle_id'], axis=1).values
        tr_y = group[['time_to_potential_event', 'class_label']].values
        
        train_X.append(tr_x)
        train_Y.append(tr_y)

    # Pad sequences to max_length with -1
    train_X = pad_sequences(train_X, maxlen=max_length, padding='post', dtype='float32', value=-1.0)
    train_Y = pad_sequences(train_Y, maxlen=max_length, padding='post', dtype='float32', value=-1.0)
    
    # Reshape the padded sequences back to the original feature dimensions
    train_X = [x for x in train_X]  # Each x is now padded to max_length
    train_Y = [y for y in train_Y]  # Each y is now padded to max_length

    # Create stratification array based on presence of class_label 4
    stratify_array = np.array([1 if (x[:, 1] == 4).any() else 0 for x in train_X])
    
    # First split: 70% vs 30%
    X_temp, X_test, y_temp, y_test, strat_temp, strat_test = train_test_split(
        train_X, train_Y, stratify_array, 
        test_size=0.15, 
        random_state=42, 
        stratify=stratify_array
    )

    # Second split: Split the 85% into ~82.35% (70/85) and ~17.65% (15/85)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.1765,  # 15/85 to get overall 15% validation set
        random_state=42,
        stratify=strat_temp
    )

    print(f"\nTraining set size: {len(y_train)}")
    print(f"Validation set size: {len(y_val)}")
    print(f"Test set size: {len(y_test)}")

    # Verify stratification
    print("\nClass 4 distribution:")
    print(f"Training set: {sum(1 for x in y_train if (x[:, 1] == 4).any()) / len(y_train):.2%}")
    print(f"Validation set: {sum(1 for x in y_val if (x[:, 1] == 4).any()) / len(y_val):.2%}")
    print(f"Test set: {sum(1 for x in y_test if (x[:, 1] == 4).any()) / len(y_test):.2%}")

    return X_train, y_train, X_val, y_val, X_test, y_test

   