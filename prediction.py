import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from src.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification
import csv 
from tqdm import tqdm 
import jsonlines 

model_name_or_path = "/lustre/scratch/client/vinai/users/linhnt140/zero-shot-tts/preprocess_audio/soxan/out_large/checkpoint-1500"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate

# for wav2vec
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

# for hubert
# model = HubertForSpeechClassification.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Label": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return outputs



# with open("../test.csv", "r") as csv_file:
#     reader = csv.reader(csv_file, delimiter='\t')
#     i = 0
#     for row in reader:
#         if i == 0:
#             i+=1
#             continue
#         print(row)
#         outputs = predict(row[0], sampling_rate)
#         print(outputs)
#         print("********************")

f = open("/lustre/scratch/client/vinai/users/thivt1/code/oneshot/artifacts/step14_tone_norm_transcript_no_multispeaker.txt", "r")
list_lines = f.read().split("\n")
f.close()
fw = jsonlines.open("data_stories_large_model.jsonl", "w")
for line in tqdm(list_lines):
    line = line.strip()
    if len(line) == 0:
        continue
    split_line = line.split("|")
    assert len(split_line) == 4
    output_write = {"path": "/lustre/scratch/client/vinai/users/thivt1/code/oneshot/" + split_line[0], "transcript": split_line[1], "speaker": split_line[2], "duration": float(split_line[3])}
    dialects = predict(output_write["path"], sampling_rate)
    output_write["dialect"] = max(dialects, key=lambda x: x['Score'])
    fw.write(output_write)
fw.close()





