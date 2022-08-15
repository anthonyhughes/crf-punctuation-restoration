import pandas as pd

from constants import DATA_ROOT

TEST_READ_PATH = DATA_ROOT + 'TED_TEST.csv'


def read_train_data():
    return pd.read_csv(TEST_READ_PATH)


if __name__ == '__main__':
    data = read_train_data()
