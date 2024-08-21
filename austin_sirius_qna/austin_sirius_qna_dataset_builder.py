from typing import Iterator, Tuple, Any, Dict
import os
import json
import tensorflow_datasets as tfds

from utils.utils import (MAX_NUM_TRAJS,
                         dataset2path, 
                         get_simple_dataset_name,
                         get_question_id, 
                         load_all_trajectories, 
                         convert_nested_dict_to_numpy)


class AustinSiriusQna(tfds.core.GeneratorBasedBuilder):
    """
    DatasetBuilder for Austin Sirius with added QnA Pairs.
    """
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    def __init__(self, *args, **kwargs):
        self.oxe_dataset_name = "austin_sirius_dataset_converted_externally_to_rlds"
        super().__init__(*args, **kwargs)

        self.train_split = "train"
        self.val_split = "val"
        self.num_train_episodes = MAX_NUM_TRAJS
        self.num_val_episodes = 2
        self.simple_dataset_name = get_simple_dataset_name(self.oxe_dataset_name)
        
        home_dir = os.path.expanduser("~")
        self.qna_data_dir = os.path.join(home_dir, "work/mmfm/oxe_qna_data_setup/oxe_qna_data/seen", self.simple_dataset_name)

    def _info(self) -> tfds.core.DatasetInfo:
        """
        Returns the dataset metadata.
        """
        builder = tfds.builder_from_directory(builder_dir=dataset2path(self.oxe_dataset_name))
        original_features = dict(builder.info.features)

        # Extract the 'steps' key's features and convert it to a dictionary
        steps_features = dict(original_features['steps'])

        # Define a new feature for question-answer pairs
        qna_features = {'question_answer_pairs': tfds.features.Sequence(
            tfds.features.FeaturesDict({
                'question_ID': tfds.features.Text(),
                'question': tfds.features.Text(),
                'answer': tfds.features.Text()}))}
        steps_features.update(qna_features)
    
        # Update the 'steps' key in the original features
        original_features['steps'] = tfds.features.Dataset(steps_features)
        final_features = tfds.features.FeaturesDict(original_features)
        return self.dataset_info_from_configs(features=final_features)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """
        Define data splits.
        """
        train_episodes = load_all_trajectories(self.oxe_dataset_name, self.train_split, self.num_train_episodes)
        val_episodes = load_all_trajectories(self.oxe_dataset_name, self.val_split, self.num_val_episodes)

        return {
            self.train_split: self._generate_examples(self.train_split, train_episodes),
            # self.val_split: self._generate_examples(self.val_split, val_episodes),
        }

    def _generate_examples(self, split, episodes) -> Iterator[Tuple[str, Any]]:
        """
        Generator of examples for each split.
        """
        def _parse_example(split, episode_idx, episode):
            # Extract QnA data
            qna_file_name = f"{split}-{episode_idx}-qna-data.json"
            qna_file_path = os.path.join(self.qna_data_dir, qna_file_name)

            if not os.path.exists(qna_file_path):
                raise FileNotFoundError(f"No such file: {qna_file_path}")
            
            with open(qna_file_path, 'r') as file:
                qna_data = json.load(file)

            assert len(qna_data) == len(episode['steps']), "Number of QnAs should be equal to the number of frames!"   
            
            steps = []
            for frame, step in enumerate(episode['steps']):
                # Modify the 'step' dictionary so all its leaf nodes are
                # numpy arrays or strings and not tensors
                step_numpy = convert_nested_dict_to_numpy(step)

                # Add QnA information for each frame
                qna: Dict = qna_data[str(frame)]
                qna_pairs = []
                for question, answer in qna.items():
                    question_id = get_question_id(question)
                    qna_pairs.append({"question_ID": question_id, 
                                      "question": question, 
                                      "answer": answer})
                step_numpy["question_answer_pairs"] = qna_pairs
                steps.append(step_numpy)

            # create output data sample
            file_path = episode['episode_metadata']['file_path'].numpy().decode()
            output = {
                'steps': steps,
                'episode_metadata': {
                    'file_path': file_path
                }
            }

            return file_path, output

        for episode_idx, episode in enumerate(episodes):
            yield _parse_example(split, episode_idx, episode)