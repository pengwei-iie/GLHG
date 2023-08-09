# coding=utf-8

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputters.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader

tokenizer_gpt = GPT2Tokenizer.from_pretrained('/data/pretrained_models/comet-distill-tokenizer')
# fixme cuda
model_gpt = GPT2LMHeadModel.from_pretrained('/data/pretrained_models/comet-distill-high').cuda()
# comet_type = ['xWant', 'xReact']
comet_type = ['xIntent']
react_type = ['xReact']


class Inputter(object):
    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features

        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader

        # valid
        self.valid_dataloader = DynamicBatchingLoader

        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


# basic utils
class InputFeatures(object):
    def __init__(
            self,
            input_ids,
            decoder_input_ids, labels,
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels

        self.input_len = self.input_length + self.decoder_input_length


def _transfer_to_comet(str, comet_type):
    return_str = []
    # max_len = len(max(str).split()) + 6
    for type in comet_type:
        tmp = "<head> " + str + " </head> <relation> " + type + " </relation> [GEN] "
        # new_want = tmp if len(tmp.split()) == max_len else tmp + "[PAD] " * (max_len-len(tmp.split()))
        return_str.append(tmp)
    return return_str


def featurize(
        bos, eos,
        context, max_input_length,
        response, max_decoder_input_length,
):
    context = [c + [eos] for c in context]
    input_ids = sum(context, [])[:-1]
    input_ids = input_ids[-max_input_length:]

    labels = (response + [eos])[:max_decoder_input_length]
    decoder_input_ids = [bos] + labels[:-1]

    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]

    return InputFeatures(
        input_ids,
        decoder_input_ids, labels,
    )


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))

    dialog = data['dialog']
    inputs = []
    context = []

    for i in range(len(dialog)):
        text = _norm(dialog[i]['text'])
        text = process(text)

        if i > 0 and dialog[i]['speaker'] == 'sys':
            str_last = " ".join(toker.convert_ids_to_tokens(last_sentence_ids))
            comet_context = context.copy()
            # begin comet
            comet_str = _transfer_to_comet(str_last, comet_type)[0]
            inp2_xWant = tokenizer_gpt(comet_str)['input_ids']
            # fixme cuda
            tmp = model_gpt.generate(input_ids=torch.tensor(inp2_xWant).view(1, -1).cuda(), num_beams=3,
                                     max_length=8 + len(inp2_xWant), num_return_sequences=1, pad_token_id=50256)
            gen = tokenizer_gpt.decode(tmp.tolist()[0])
            intention_ = gen[gen.index(']') + 1:]
            intention_ = intention_.split('.')[0] + "."
            comet_ids = process(intention_)
            comet_context = comet_context + [comet_ids]
            # begin react
            react_str = _transfer_to_comet(str_last, react_type)[0]
            # fixme cuda
            inp2_xReact = tokenizer_gpt(react_str)['input_ids']
            tmp_React = model_gpt.generate(input_ids=torch.tensor(inp2_xReact).view(1, -1).cuda(), num_beams=3,
                                           max_length=8 + len(inp2_xReact), num_return_sequences=1, pad_token_id=50256)
            gen_react = tokenizer_gpt.decode(tmp_React.tolist()[0])
            intention_react = gen_react[gen_react.index(']') + 1:]
            intention_react = intention_react.split('.')[0] + "."
            react_ids = process(intention_react)
            comet_context = comet_context + [react_ids]
            res = {
                'context': context.copy(),
                'response': text,
                'comet_ids': comet_ids,
                'comet_text': intention_,
                'React_ids': react_ids,
                'React_text': intention_react,
                'comet_context': comet_context,
            }

            # res = {
            #     'context': context.copy(),
            #     'response': text,
            #     'comet_ids': comet_ids,
            #     'comet_text': intention_,
            #     'comet_context': comet_context,
            # }

            inputs.append(res)

        context = context + [text]
        last_sentence_ids = text

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos,
            ipt['comet_context'], max_input_length,
            ipt['response'], max_decoder_input_length,
        )
        features.append(feat)
    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, infer=False):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        bos = toker.bos_token_id
        if bos is None:
            bos = toker.cls_token_id
            assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = toker.eos_token_id
        if eos is None:
            eos = toker.sep_token_id
            assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features],
                                 batch_first=True, padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
                                      batch_first=True, padding_value=0.)
        input_length = torch.tensor([f.input_length for f in features], dtype=torch.long)

        if not infer:
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                                             batch_first=True, padding_value=pad)
            labels = pad_sequence([torch.tensor(f.labels, dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=-100)
        else:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
            labels = None

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'input_length': input_length,

            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }

        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """

    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()

            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []

            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch

        except StopIteration:
            pass

    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


# for inference
def prepare_infer_batch(features, toker, interact=None):
    res = FeatureDataset.collate(features, toker, True)

    res['batch_size'] = res['input_ids'].size(0)

    return res


def get_infer_batch(infer_input_file, toker, **kwargs):
    assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
    infer_batch_size = kwargs.get('infer_batch_size')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()

    features = []
    sample_ids = []
    posts = []
    references = []
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring"):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, **kwargs)
        tmp_features = convert_inputs_to_features(inputs, toker, **kwargs)
        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            posts.append(toker.decode(ipt['context'][-1]))
            references.append(toker.decode(ipt['response']))
            sample_ids.append(sample_id)

            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker), posts, references, sample_ids
                features = []
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references, sample_ids
