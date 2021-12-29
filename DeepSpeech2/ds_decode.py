import argparse

import numpy as np
import pandas as pd
import torch
import json

from tqdm import tqdm
import decoder

from decoder import GreedyDecoder
from torch.autograd import Variable
from opts import add_decoder_args, add_inference_args
from utils import load_model

from data.data_loader import SpectrogramParser


def decode_results(model, decoded_output, decoded_offsets):
    results = {
        "output": [],
        # "_meta": {
        #     "acoustic_model": {
        #         "name": os.path.basename(args.model_path)
        #     },
        #     "language_model": {
        #         "name": os.path.basename(args.lm_path) if args.lm_path else None,
        #     },
        #     "decoder": {
        #         "lm": args.lm_path is not None,
        #         "alpha": args.alpha if args.lm_path is not None else None,
        #         "beta": args.beta if args.lm_path is not None else None,
        #         "type": args.decoder,
        #     }
        # }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results

def transcribe(audio_path, parser, model, device):
    c_max_rows = 555

    print(audio_path)
    spect = parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    print(input_sizes)
    out = model(Variable(spect, volatile=True))
    print(model._labels)
    test1, test2 = decoder.GreedyDecoder(model._labels).decode(out)
    print("****************", )
    print(json.dumps(decode_results(model, test1, test2)))
    out = np.squeeze(out.data.numpy())
    print("****************************")
    print(out.shape)
    out = np.pad(out, ((0, c_max_rows - out.shape[0]), (0, 32 - out.shape[1])), 'constant')
    print(out.shape)
    return out


# load model
parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--audio-path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
# parser.add_argument('--model-path', default='models/librispeech_pretrained.pth')
parser = add_decoder_args(parser)
args = parser.parse_args([])
device = torch.device("cuda" if args.cuda else "cpu")

model = load_model(device, args.model_path, args.cuda)
parser = SpectrogramParser(model._audio_conf, normalize=True)

lang = "Tamil"
base_path = '/home/isham/Projects/MSc/Speech_Intent/'

audio_folder = 'data/' + lang + '_Dataset/audio_files/'
# read csv from csv
data = pd.read_csv(base_path + 'data/' + lang + '_Dataset/' + lang + '_Data.csv')
file_names = data['audio_file']
file_names = file_names.apply(lambda x: base_path + audio_folder + str(x))

# audio_folder = 'data/formatted_data/'
# # read csv from csv
# data = pd.read_csv(base_path + 'data/formatted_data_v2.csv')
# file_names = data['filename']
# file_names = file_names.apply(lambda x: base_path + audio_folder + str(x))


file_names = file_names.values


ds2_features = []

for f in tqdm(file_names):
    feature = transcribe(f, parser, model, device)
    ds2_features.append(feature)

ds2_features = np.array(ds2_features)
print(ds2_features.shape)

np.save(base_path + 'data/' + lang + '_Dataset/ds2_decode_' + lang + '_data_v2', ds2_features)
