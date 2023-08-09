import torch
# from allennlp.predictors import SemanticRoleLabelerPredictor
import json
# from allennlp.predictors import Predictor
# from nltk.tokenize import sent_tokenize
import pandas as pd
import spacy
import re
from tqdm import tqdm
import pickle
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from keras.preprocessing.sequence import pad_sequences
NUM_RETURN = 3


# class SRL():
#     def __init__(self):
#         #use the model from allennlp for SRL.
#         # self.predictor = SemanticRoleLabelerPredictor.from_path("/data1/data-xlx/pytorch-pretrain-lm/elmo-model-srl")
#         self.predictor = Predictor.from_path("./models/")
#         self.predictor._model = self.predictor._model.cuda()
#
#     def post_allen(self, sentence):
#         results = self.predictor.predict_batch_json(sentence)
#         return results
#
#     def annoatate(self, seg_file, batch):
#         with open(seg_file, 'r', encoding='utf-8') as data_file:
#             reader = pd.read_csv(data_file)
#             # next (reader)
#             count = 0
#             # all_data = []
#             # for line in reader:
#             #     all_data.append(line[0])
#             #     all_data.append(line[1])
#             #     # break
#             # print(all_data[0])
#
#             # new_file.write ('storyid_linenum\ttopic\tbehavior\tsrl_annotation\n')
#             # generate batch for SRL
#             # data_chunk = [all_data[i:i + batch] for i in range(0, len(all_data), batch)]
#             # index_of_sample = [2,3,4,5,6]
#             for index in tqdm(range(len(reader))):
#                 # data_sent 一个字符串的列表，text是一个dict的列表
#                 # for index in index_of_sample:
#                 data_sent = sent_tokenize(reader.iloc[index]['U1'])
#                 text = [{"sentence": sent} for sent in data_sent]
#                 text_tags = self.post_allen(text)
#
#                 # event level
#                 line = ""
#                 for raw_data, tag in zip (data_sent, text_tags):
#                     nlp = spacy.load ('en_core_web_sm')
#                     doc = nlp(raw_data)
#                     for token in [token for token in doc]:
#                         # find the ROOT token in dependency tree
#                         if token.dep_=='ROOT':
#                             srl_taggings = tag['verbs']
#                             for srl_tagging in srl_taggings:
#                                 if srl_tagging['verb']==token.text:
#                                     # id = 'id__sent' + str(i+1)
#                                     # result_list = re.findall("(?<=\[).*?(?=\])", srl_tagging['description'])
#                                     if len(line) == 0:
#                                         line = srl_tagging['description']
#                                     else:
#                                         line = line + " " + srl_tagging['description']
#                                     # line = line + "\n"
#                                     # new_file.write(line)
#                     # reader.at[index] = pd.Series([line], index=["Extracted_U1"])
#                 reader.loc[index, "Extracted_U1"] = line
#             # reader.to_csv(data_path + "_extractd.csv", index=False, header=True)
#
#                 # dela with u2
#                 data_sent = sent_tokenize(reader.iloc[index]['U2'])
#                 text = [{"sentence": sent} for sent in data_sent]
#                 text_tags = self.post_allen(text)
#
#                 # event level
#                 line = ""
#                 for raw_data, tag in zip(data_sent, text_tags):
#                     nlp = spacy.load('en_core_web_sm')
#                     doc = nlp(raw_data)
#                     for token in [token for token in doc]:
#                         # find the ROOT token in dependency tree
#                         if token.dep_ == 'ROOT':
#                             srl_taggings = tag['verbs']
#                             for srl_tagging in srl_taggings:
#                                 if srl_tagging['verb'] == token.text:
#                                     # id = 'id__sent' + str(i+1)
#                                     # result_list = re.findall("(?<=\[).*?(?=\])", srl_tagging['description'])
#                                     if len(line) == 0:
#                                         line = srl_tagging['description']
#                                     else:
#                                         line = line + " " + srl_tagging['description']
#                                     # line = line + "\n"
#                                     # new_file.write(line)
#                 reader.loc[index, "Extracted_U2"] = line
#             # reader.to_csv(data_path + "_extractd.csv", index=False, header=True)


# class InputFeatures_blender(object):
#     def __init__(self, encoder_feature, decoder_feature, intention_text):
#         self.conv_id = encoder_feature.conv_id
#         self.input_ids = encoder_feature.input_ids
#         self.position_ids = encoder_feature.position_ids
#         self.token_type_ids = encoder_feature.token_type_ids
#         self.role_ids = encoder_feature.role_ids
#         self.lm_labels = encoder_feature.lm_labels
#         self.cls_position = encoder_feature.cls_position
#         self.cls_label = encoder_feature.cls_label
#         self.strategy_ids = encoder_feature.strategy_ids
#         self.decoder_input_ids = decoder_feature.input_ids
#         self.decoder_position_ids = decoder_feature.position_ids
#         self.decoder_token_type_ids = decoder_feature.token_type_ids
#         self.decoder_role_ids = decoder_feature.role_ids
#         self.decoder_lm_labels = decoder_feature.lm_labels
#         self.decoder_cls_position = decoder_feature.cls_position
#         self.decoder_cls_label = decoder_feature.cls_label
#         self.decoder_strategy_ids = decoder_feature.strategy_ids
#         self.intention_text = intention_text


def _transfer_to_comet(str, comet_type):
    return_str = []
    # max_len = len(max(str).split()) + 6
    for text in str:
        tmp = "<head> " + text + " </head> <relation> " + comet_type + " </relation> [GEN] "
        # new_want = tmp if len(tmp.split()) == max_len else tmp + "[PAD] " * (max_len-len(tmp.split()))
        return_str.append(tmp)
    return return_str

def _padding(ids):
    return_str = []
    max_len = len(max(ids))
    for id in ids:
        new_want = id if len(id) == max_len else id + "[PAD] " * (max_len-len(tmp))
        return_str.append(tmp)
    return return_str

# cached_features_file = 'data/pengwei/fourth_next/phy_diag/data/cached/baseline_dev_cached_lm_512intention_idstext'
# with open(cached_features_file + "text", "wb") as outfile:
#     pickle.load(outfile)

str = ['Hi. I had problems in the past and somehow right now.',
       'I loved physics and I decided to attend university, but soon after I slowly lost my interest and became unmotivated.',
       'so I started to skipping the classes and failing.',
       'Yes lots of stress and pressure. I did not, because we didn\'t have such things or at least I wasn\'t aware.']
comet_type = 'xEffect'
comet_str = _transfer_to_comet(str, comet_type)
time_start = time.time()
tokenizer = GPT2Tokenizer.from_pretrained('/data/pretrained_models/comet-distill-tokenizer')
model = GPT2LMHeadModel.from_pretrained('/data/pretrained_models/comet-distill-low').cuda()
# case2 = ["<head> Amy eats an apple. </head> <relation> xWant </relation> [GEN]",
#          "<head> Amy is hungry and go to the shopping mall. </head> <relation> xWant </relation> [GEN] "]
inp2 = tokenizer(comet_str)['input_ids']
padding = pad_sequences(inp2, maxlen=max([len(i) for i in inp2]), padding='post', value=tokenizer(["[PAD]"])["input_ids"][0][0]).tolist()
ids = torch.tensor(padding).cuda()
# tmp = model.generate(input_ids=torch.tensor(inp).view(1,-1), num_beams=5,
#                        max_length=30, num_return_sequences=NUM_RETURN)
tmp = model.generate(input_ids=ids, num_beams=5, max_length=40, num_return_sequences=NUM_RETURN)
# print(tmp)
# print(tmp.tolist())
# print(type(tmp))
# print(tmp.size())
print(time.time()-time_start)
for i in range(NUM_RETURN*ids.size()[0]):
    gen = tokenizer.decode(tmp.tolist()[i])
    print(gen)
