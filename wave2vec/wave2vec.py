import torch
import fairseq
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import shlex



cp_path = '../models/wav2vec_large.pt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

signal, _ = sf.read("audio1520623611766_1_1.wav")
signal = torch.tensor(signal, device=device).float().unsqueeze(0)
wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)
print(c.shape)


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

maxShape = (0, 256)
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

    # phonemes = model.decode_phonemes()
    z = model.feature_extractor(signal)
    c = model.feature_aggregator(z)
    out = c.detach().numpy()
    maxShape = max(out.shape, maxShape)

    # out = phonemes.detach().numpy()
    print(out.shape)
    print(out)
    # out = np.pad(out, ((0, c_max_rows - out.shape[0]), (0, 0)), 'constant')
    # print(out.shape)
    # print(out)
    # ds2_features.append(out)
    # ds2_features.append(phonemes.detach().numpy())
    # print((phonemes.detach().numpy()).shape)
    # print(out.shape)

# ds2_features = np.array(ds2_features)
# print(ds2_features.shape)
# print(ds2_features)

print(maxShape)
np.save(base_path + 'data/' + lang + '_Dataset/wav2vec_decode_' + lang + '_data_v2', ds2_features)
