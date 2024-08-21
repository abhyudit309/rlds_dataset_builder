from typing import Optional, Union, List, Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import gc

# maximum number of trajectories extracted from the Open-X dataset
# same as in oxe_qna_data_setup/prepare_qna_for_oxe.py
MAX_NUM_TRAJS = 500


def dataset2path(dataset_name: str) -> str:
    """
    Helper function that generates the correct path
    with the version for each dataset from Open-X.

    Taken directly from configs/dataset_configs.py.
    """
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'


def get_simple_dataset_name(dataset_name: str) -> str:
    """
    Helper function that converts the registered tf dataset name
    to its simple form.

    Taken directly from configs/dataset_configs.py.
    """
    if "rlds" in dataset_name:
        if "dataset" in dataset_name:
            simple_name = dataset_name.split("_dataset")[0]
        else:
            simple_name = dataset_name.split("_converted")[0]
        return simple_name
    else:
        return dataset_name


def get_question_id(question: str) -> str:
    """
    Helper function that obtains the question ID of a question defined in
    oxe_qna_data_setup/prepare_qna_for_oxe.py.
    """
    if question == "What is the next action for the robot end-effector?":
        return "R1"
    elif question == "What is the next pose of the robot end-effector?":
        return "R2"
    elif question == "What is the status of the robot gripper?":
        return "R3"
    elif question == "What are the objects in the image?":
        return "S1"
    elif question.startswith("What is the location of"):
        return "S2"
    elif question.startswith("What object is in the location"):
        return "S3"
    elif question == "What objects are involved in the task?":
        return "S4"
    elif question == "Describe the image.":
        return "S5"
    elif question == "What action is the robot performing and why?":
        return "S6"
    else:
        raise NotImplementedError


def load_all_trajectories(dataset_name: str, 
                          split: str,
                          num_trajs: Optional[int] = None
                          ) -> List[Dict[Dict, Any]]:
    """
    Loads all/some trajectories from one of the Open-X Franka datasets.

    This function loads all/some trajectories based on the specified 
    dataset and split.

    Taken with a few edits from utils/loaders.py. Basically, this does NOT
    standardize or transform the trajectory in any way.

    Args:
        dataset_name (str): The name of the dataset to load from. 
        split (str): Split to load data from. It can be "train" or "val".
        num_trajs (Optional[int]): Number of trajectories to load. 
        If None, it loads all trajectories.

    Returns:
        trajs (List[Dict[Dict, Any]]): A dictionary containing all the trajectories. 
        Images, language instructions and ground truth actions can be accessed from this.

    Raises:
        AssertionError: If 'split' is not "train" or "val".
    """
    assert split in ["train", "val"], "Invalid split specified!"
    builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))

    if "val" not in builder.info.splits:
        if split == "train":
            actual_split = "train[:95%]"
        else:
            actual_split = "train[95%:]"
    else:
        actual_split = split

    dataset = builder.as_dataset(split=actual_split)

    # Take some or all trajectories
    if num_trajs is None:
        num_trajs = len(dataset)
    else:
        num_trajs = min(num_trajs, len(dataset))
    trajs = [traj for traj in dataset.take(num_trajs)]

    # Clear memory
    del dataset
    gc.collect()

    return trajs


def convert_tensor_to_numpy_or_string(tensor: tf.Tensor) -> Union[np.ndarray, str]:
    """
    Converts a TensorFlow tensor to a NumPy array or string, whatever 
    is applicable
    
    Args:
        tensor (tf.Tensor): The input TensorFlow tensor.
    
    Returns:
        np_array_or_str (Union[np.ndarray, str]): The converted NumPy array or string.
    """
    if tensor.dtype == tf.string:
        np_array_or_str = tensor.numpy().decode()
    else:
        np_array_or_str = tensor.numpy()
    return np_array_or_str


def convert_nested_dict_to_numpy(nested_dict: Dict) -> Dict:
    """
    Recursively converts a nested dictionary of TensorFlow tensors to NumPy arrays.
    
    Args:
        nested_dict (Dict): The input nested dictionary with TensorFlow tensors as leaf values.
    
    Returns:
        converted_dict (Dict): A new nested dictionary with NumPy arrays as leaf values.
    """
    converted_dict = {}
    
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            converted_dict[key] = convert_nested_dict_to_numpy(value)
        elif isinstance(value, tf.Tensor):
            converted_dict[key] = convert_tensor_to_numpy_or_string(value)
        else:
            converted_dict[key] = value
    
    return converted_dict