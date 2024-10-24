# import tensorflow as tf
import pandas as pd
import os

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
        return -1 #No failure reported, but within 48 time steps from the end of the study, don't know if it will fail or not
    
def add_class_labels(tte, readouts):
    # Join the readouts and the time to event data
    df = pd.merge(readouts, tte, on = 'vehicle_id', how='left').copy()
    #Calculate the time to a failure event
    df['time_to_potential_event'] = df['length_of_study_time_step'] - df['time_step']
    df['class_label'] = df.apply(get_class_label, axis=1)
    return df

if __name__ == '__main__':
    print()

    print(os.getcwd())

    dataset_dir = "/workspaces/datasets/scania"
    print(sorted(os.listdir(dataset_dir)))

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

    print([train_tte.columns, train_readouts.columns])
    print([val_tte.columns, val_readouts.columns])
    print([test_tte.columns, test_readouts.columns])


    df_train = add_class_labels(train_tte, train_readouts)
    df_val = add_class_labels(val_tte, val_readouts)
    df_test = add_class_labels(test_tte, test_readouts)

    print(df_train.head())
    print(df_val.head())
    print(df_test.head())


    







