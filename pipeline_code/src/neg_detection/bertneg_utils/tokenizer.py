import pandas as pd
import pickle
from transformers import BertTokenizer, AutoTokenizer
import torch
from torch.utils.data import TensorDataset, random_split
from random import randint
from spacy.training import  offsets_to_biluo_tags as biluo_tags_from_offsets
from spacy.lang.en import English

class Tokenizer():
    def __init__(self, df, sample, span, id, bert_input_size = 256, bert_type = "spanbert",is_bioscope=False):
        self._sample = sample
        self._span = span
        self._id = id

        if bert_type == "bert":
            self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

        self._bert_input_size = bert_input_size
        self._df = df

        self.nlp = English()

        if is_bioscope:
            self._df = self._df.apply(lambda row: self._create_iob(row, self.nlp,is_bioscope), axis=1)

    def get_tokenizer(self):
        return self._tokenizer

    def _biluo_to_iob(self, biluo):
        return [t.replace('U', 'B', 1).replace('L', 'I', 1) for t in biluo]

    def _create_final(self,df):
        new_df = pd.DataFrame({})
        for idx, row in df.iterrows():
            for (s,e) in row.correct_intervals:
                new_df = new_df.append(pd.Series({
                    "text_id": idx,
                    "text": row.tweet,
                    "begin": int(s),
                    "end": int(e),
                    "type": "ADR"
                }),ignore_index=True)
        return new_df

    def _create_iob(self,row,nlp,is_bioscope):
        row['gold_labels'] = [s.replace("-ADR","") for s in row['iob_tags']]
        return row
    
    def get_df(self):
        return self._df

    def tokenize_evaluation(self):
        df = self._df
        new_df = pd.DataFrame({})

        for idz,row in df.iterrows():
            doc = self.nlp(row.tweet)

            CLS = self._tokenizer.cls_token
            PAD = self._tokenizer.pad_token
            SEP = self._tokenizer.sep_token

            tokens = [str(token) for token in doc]
            extended_tokens = tokens

            extended_tokens = [CLS] + extended_tokens + [SEP]
            bert_t = []
            for word in extended_tokens:
                bert_t.extend(self._tokenizer.tokenize(word))

            tokens_with_pad = self._add_padding(bert_t)


            mask            = self._create_mask(bert_t)

            row['spacy_tokens'] = tokens
            row['bert_tokens'] = bert_t#[self._tokenizer.tokenize(word) for word in tokens_with_pad]
            row['bert_ids'] = [self._tokenizer.convert_tokens_to_ids(token) for token in bert_t]
            row['mask'] = mask
            new_df = new_df.append(row)

        self._df = new_df

    def tokenize_training(self):
        df = self._df
        new_df = pd.DataFrame({})
        i =0
        for idz, row in df.iterrows():
            tokens, labels = row.spacy_tokens, row.gold_labels

            extended_tokens = []
            extended_labels = []

            for (word, label) in zip(tokens, labels):
            
                tokenized_word = self._tokenizer.tokenize(word)
                n_subwords = len(tokenized_word)

                extended_tokens.extend(tokenized_word)

                suffix = ''
                if len(label) > 1:
                    suffix = label[1:]

                if label[0] == 'B':
                    extended_labels.extend(['B'] + ['I'] * (n_subwords - 1))

                else:
                    extended_labels.extend([label] * n_subwords)


            if len(extended_labels) != len(extended_tokens):
                extended_labels = extended_labels[:len(extended_labels)-(len(extended_labels) - len(extended_tokens))]



            CLS = self._tokenizer.cls_token
            PAD = self._tokenizer.pad_token
            SEP = self._tokenizer.sep_token

            # adding special tokens
            extended_tokens = [CLS] + extended_tokens + [SEP]
            extended_labels = ["O"] + extended_labels + ["O"]

            tokens_with_pad = self._add_padding(extended_tokens)
            mask            = self._create_mask(tokens_with_pad)
            final_label     = self._build_final_label(extended_labels)
            numerical_label = self._build_final_numerical_label(final_label)
            row['bert_tokens'] = tokens_with_pad
            row['spacy_tokens'] = tokens
            row['bert_ids'] = [self._tokenizer.convert_tokens_to_ids(token) for token in tokens_with_pad]
            row['mask'] = mask
            row['bert_gold_labels'] = final_label
            row['bert_gold_labels_num'] = numerical_label
            new_df = new_df.append(row)

        self._df = new_df

    def _add_padding(self, tokens):
        tokens.extend([self._tokenizer.pad_token for _ in range(self._bert_input_size - len(tokens))])
        return tokens

    def _create_mask(self,tokens):
        return [0 if token == self._tokenizer.pad_token else 1 for token in tokens]

    def _build_final_label(self,labels):
        labels.extend(["O" for _ in range(self._bert_input_size - len(labels))])
        return labels

    def _build_final_numerical_label(self, labels):
        iob_to_n = {
            'O': 0,
            'I': 1,
            'B': 2
        }
        res = [iob_to_n[label] for label in labels]
        return res
    
    def get_complete_dataset_with_tokens(self):
        dataset = self.get_complete_dataset()
        return dataset, self._df

    def get_tokens_from_ids(self, ids):
        return [self._tokenizer.convert_ids_to_tokens(_id) for _id in ids]

    def get_augmented_df(self):
        return self._df

    def get_complete_dataset_training(self):
        ids              = torch.cat([torch.tensor([w],dtype=torch.long) for w in self._df.bert_ids.values])
        masks            = torch.cat([torch.tensor([w],dtype=torch.long) for w in self._df['mask'].values])
        labels           = torch.cat([torch.tensor([w],dtype=torch.long) for w in self._df.bert_gold_labels_num.values])
        return TensorDataset(ids, masks, labels)

    def get_complete_dataset_evaluation(self):
        ids              = torch.cat([torch.tensor([w],dtype=torch.long) for w in self._df.bert_ids.values])
        masks            = torch.cat([torch.tensor([w],dtype=torch.long) for w in self._df['mask'].values])
        return TensorDataset(ids, masks)