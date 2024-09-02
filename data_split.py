from sklearn.model_selection import train_test_split


def train_dev_test_split(data_list):

    # split the dataset into train and test
    train_data, test_data = train_test_split(data_list, test_size=0.1, random_state=5)

    train_data, dev_data = train_test_split(train_data, test_size=0.2, random_state=5)
    #print(len(train_data), len(test_data), len(dev_data))
    
    return train_data, dev_data, test_data


def main():

    with open("reviews.txt") as f:
        data = f.readlines()

    # split the data into train (70%), dev (20%) and test (10%) sets
    train_data, dev_data, test_data = train_dev_test_split(data)

    # save the datasets as txt files
    with open("train.txt", "w") as f1:
        f1.write("".join(train_data))

    with open("dev.txt", "w") as f2:
        f2.write("".join(dev_data))

    with open("test.txt", "w") as f3:
        f3.write("".join(test_data))
    
if __name__ == "__main__":
    main()