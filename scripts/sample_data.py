import pandas as pd
import os
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # define constants
    random_state = 1
    test_size = 0.2

    # read data
    # change path format if using Linux
    base_path = os.getcwd()
    path = f'{base_path}\\data\\group_proj_cleaned_set.csv'
    df = pd.read_csv(path)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # write out separate datasets
    train_df.to_csv(f'{base_path}\\data\\training_data.csv')
    val_df.to_csv(f'{base_path}\\data\\validation_data.csv')
