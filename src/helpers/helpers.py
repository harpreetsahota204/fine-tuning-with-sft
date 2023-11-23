from pathlib import Path
from typing import Optional
from datasets import Dataset
import random

def create_directory(path: Optional[Path] = None, dir_name: str = "output"):
    """
    Creates a directory at the specified path with the given directory name.
    If no path is provided, the current working directory is used.

    Parameters:
    - path (Optional[Path]): The path where the directory is to be created.
    - dir_name (str): The name of the directory to create.

    Returns:
    - Path object representing the path to the created directory.
    """
    # Use the current working directory if no path is provided
    working_dir = path if path is not None else Path('./')

    # Define the output directory path by joining paths
    output_directory = working_dir / dir_name

    # Create the directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory

def select_random_rows(dataset: Dataset,
                       num_samples: int,
                       seed: int = 42) -> Dataset:
    """
    Select a random subset of rows from a HuggingFace dataset object.

    Args:
        dataset (Dataset): The dataset to sample from.
        num_samples (int): The number of random samples to select.
        seed (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
        Dataset: A new dataset object containing the randomly selected rows.
    """
    if num_samples > len(dataset):
        raise ValueError("num_samples is greater than the number of rows in the dataset.")

    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)

    random_indices = random.sample(range(len(dataset)), num_samples)
    random_rows = dataset.select(random_indices)

    return random_rows
