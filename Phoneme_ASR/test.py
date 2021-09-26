import data
import models
import soundfile as sf
import torch

import numpy as np
import pandas as pd
import subprocess
import shlex

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg")
_, _, _ = data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device)) # load trained model

# signal, _ = sf.read("audio1520623611766_1_1.wav")
# signal = torch.tensor(signal, device=device).float().unsqueeze(0)
#
# model.decode_intents(signal)
# phonemes = model.decode_phonemes(signal)
# intents = model.decode_intents(signal)
# print(intents)
# print(phonemes.size())
# print(phonemes)
# print(phonemes)
# print(torch.sum(phonemes, dim=0).size())
# print(torch.sum(phonemes, dim=0))
# print(torch.sum(phonemes, dim=1).size())
# print(torch.sum(phonemes, dim=1))
# values, indices = torch.max(phonemes, dim=1)
# print(indices.size())
# print(indices)

# print(model.pretrained_model.compute_posteriors(signal)[0].max(2)[1])

# with open("experiments/no_unfreezing/pretraining/phonemes_bkp_ipa.txt", "r") as f:
#     phonemes = f.readlines()
#     phonemes = [p.strip() for p in phonemes]
#     indices = model.pretrained_model.compute_posteriors(signal)[0].max(2)[1]
#     print(indices)
#     ipa_list = [phonemes[index.item()] for index in indices[0]]
#     print(ipa_list)
#     print(''.join(ipa_list))


lang = "Tamil"
# lang = "Sinhala"

base_path = '/home/isham/Projects/MSc/Speech_Intent/'

# audio_folder = 'data/formatted_data/'
# # read csv from csv
# data = pd.read_csv(base_path + 'data/formatted_data_v2.csv')
# file_names = data['filename']
# file_names = file_names.apply(lambda x: base_path + audio_folder + str(x))

audio_folder = 'data/' + lang + '_Dataset/audio_files/'
# read csv from csv
data = pd.read_csv(base_path + 'data/' + lang + '_Dataset/' + lang + '_Data.csv')
file_names = data['audio_file']
file_names = file_names.apply(lambda x: base_path + audio_folder + str(x))

file_names = file_names.values
ds2_features = []

for f in tqdm(file_names):
    c_max_rows = 256
    signal, _ = sf.read(f)
    signal = torch.tensor(signal, device=device).float().unsqueeze(0)

    if len(signal.size()) > 2:
        # print('Stereo')
        command = 'ffmpeg -i ' + f + \
                  ' -ac 1 -ab 256000 -ar 16000 ' + f + ' -y'
        subprocess.run(shlex.split(command), stdout=subprocess.PIPE)

        signal, _ = sf.read(f)
        signal = torch.tensor(signal, device=device).float().unsqueeze(0)

    print('shape: ', signal.size(), ' len: ', len(signal.size()))

    phonemes = model.decode_phonemes(signal)
    out = phonemes.detach().numpy()
    print("***********************")
    print(phonemes.shape)
    print(out.shape)
    print(out)
    out = np.pad(out, ((0, c_max_rows - out.shape[0]), (0, 0)), 'constant')
    print(out.shape)
    print(out)
    ds2_features.append(out)
    # ds2_features.append(phonemes.detach().numpy())
    # print((phonemes.detach().numpy()).shape)
    # print(out.shape)

ds2_features = np.array(ds2_features)
# print(ds2_features.shape)
# print(ds2_features)


np.save(base_path + 'data/' + lang + '_Dataset/phoneme_decode_' + lang + '_data_v2', ds2_features)