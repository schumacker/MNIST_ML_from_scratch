import numpy as np
import matplotlib.pyplot as plt

def split_train_test(data: np.ndarray, targets: np.ndarray, rate: float=0.8, 
                  reshuffle: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly split the dataset into train (first 80%) and test (remaining 20%).
    It keeps the dataset balanced in terms of target distribution 
    (i.e., make sure that the frequency of samples per class is balanced).
    
    Parameters:
        data (np.ndarray): N-D array in which each row is a data sample.
        targets (np.ndarray): 1-D array with the target value for each feature.
        rate (float): Defines the split [rate, 1-rate] defining the 
            [train, test split]. The value should be in the range [0.0, 1.0].
    """
    
    # Make sure that the number of targets mach the number of samples
    if data.shape[0] != targets.shape[0]:
        err_msg = "Number of samples ({0}) mismatches number of targets ({1})."
        raise Exception(err_msg.format(data.shape[0], targets.shape[0]))
        
    if rate < 0.0 or rate > 1.0:
        raise Exception("Invalid split rate: {0}.".format(rate))
        
    target_values = np.unique(targets)
    full_set = np.hstack((data, targets[:, np.newaxis]))
    
    # Make sure the data selection is not biased on 
    np.random.shuffle(full_set)
    
    trainset = None
    testset = None
    
    for label in target_values:
        class_samples = full_set[np.where(full_set[:, -1] == label)]
        train_size = int(np.around(class_samples.shape[0] * rate))
        test_size = class_samples.shape[0] - train_size
        
        if trainset is None:
            trainset = class_samples[:train_size, :]
            testset = class_samples[-test_size:, :]
        else:
            trainset = np.vstack((trainset, class_samples[:train_size, :]))
            testset = np.vstack((testset, class_samples[-test_size:, :]))
       
    if reshuffle:
        np.random.shuffle(trainset)
        np.random.shuffle(testset)
        
    return (trainset, testset)

def plot_data_distribution(labels: np.ndarray, title: str=None) -> None:
    x, y = np.unique(labels, return_counts=True)  
    plt.bar(x, y)
    
    if title:
        plt.title(title)
        
    plt.xlabel("Data labels")
    plt.ylabel("Number of samples")
    plt.show()
    
def check_intersection(data1: np.ndarray, data2: np.ndarray) -> list[tuple[np.ndarray, int]]:       
    result = []
        
    for row in data1:
        count = np.all(data2 == row, axis=1).sum()
        
        if count > 0:
            result.append((row, count))
            
    return result

def remove_duplicates(dataset: np.ndarray) -> tuple[np.ndarray, int]:
    dataset, count = np.unique(dataset, axis=0, return_counts=True)
    return dataset, (count > 1).sum()