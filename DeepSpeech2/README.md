This code is obtained from the  git repo `https://github.com/SeanNaren/deepspeech.pytorch` and changed to work with the available model under release version 1.1

Dependencies
- Create a conda environment using [conda_environment.yml](conda_environment.yml) file.

ASR model
- Download ([Link](https://drive.google.com/file/d/1SpyK0SV7hqaEeJ1g7B2wdrjF5e3etSBJ/view?usp=sharing)) and place pre-trained English model in `models` folder in the current directory.

Generate Features
- Change 'ds_decode.py' appropriately to generate features from the respective audio files.