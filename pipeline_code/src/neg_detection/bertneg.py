from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers import BertModel, AdamW, BertConfig, AutoModel
from neg_detection.bertneg_utils.bioscope import Bioscope
from tqdm import tqdm
import os
from icecream import ic
import torch
import pandas as pd
import spacy
from utils.evaluator import Evaluator
from config import *
from neg_detection.bertneg_utils.tokenizer import Tokenizer
from neg_detection.bertneg_utils.detokenizer import detokenize
from neg_detection.bertneg_utils.get_predictions import get_predictions
from utils.utils import mic
class NegDetectModel(nn.Module):
    def __init__(self,bert_type):
        super(NegDetectModel, self).__init__()
        if bert_type == "spanbert":
            self.bert = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
        if bert_type == "bert":
            self.bert = BertModel.from_pretrained("bert-base-uncased")
                            #hidden_dropout_prob=0.2
        self.linear = nn.Linear(768,3)
        self.relu = nn.ReLU()

    def forward(self,x,mask):
        bert_out = self.bert(x, token_type_ids=None, attention_mask = mask)
        return self.relu(self.linear(bert_out[0]))

class BERTneg():
    def __init__(self, train = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if train:
            mic("Training...")
            self.train()
        else:
            mic("Testing...")
            self.test()

    def test(self):
        self.model = NegDetectModel(bert_type="bert")
        self.model.load_state_dict(torch.load(os.path.join(Config.PATH_OUTPUT.value, "models", "bertneg_bioscope.model")))
        self.model.to(self.device)

    def train(self):
        df_train, df_val = Bioscope("negation").get_train_val_sets()
        ## dataloader train
        tokenizer_train = Tokenizer(df=df_train,sample="sentence",span="correct_intervals",id="id",is_bioscope=True)
        tokenizer_train.tokenize_training()
        dataset_train = tokenizer_train.get_complete_dataset_training()
        
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
        ## dataloader validation
        tokenizer_val = Tokenizer(df=df_val,sample="sentence",span="correct_intervals",id="id",is_bioscope=True)
        tokenizer_val.tokenize_training()
        dataset_val = tokenizer_val.get_complete_dataset_training()
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
        
        ## model info
        
        model = NegDetectModel(bert_type="bert")
        model.to(self.device)
        optimizer  = torch.optim.AdamW(model.parameters(), lr=0.0001, eps=5e-5)
        
        df = pd.DataFrame({})
        add = True


        for epoch in tqdm(range(10)):
            model.train()
            for data in dataloader_train:
                x_ids, x_mask, x_labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

                optimizer.zero_grad()
            
                logits = model(x_ids,x_mask)

                loss = self.calculate_loss(attention_mask = x_mask, logits = logits, labels = x_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            g_logits, g_labels, g_masks = [], [], []
            with torch.no_grad():
                for data_val in dataloader_val:
                    # predictions
                    x_ids_val, x_mask_val, x_labels_val = data_val[0].to(self.device), data_val[1].to(self.device), data_val[2].to(self.device)
                    logits_val = model(x_ids_val,x_mask_val)
                    #loss = calculate_loss(attention_mask = x_mask, logits = logits, labels = x_labels)
                    ###

                    x_mask_val = x_mask_val.detach().cpu()
                    x_mask_val.squeeze().tolist()

                    x_labels_val = x_labels_val.detach().cpu()
                    logits_val = logits_val.detach().cpu()
                    logits_val = torch.argmax(torch.FloatTensor(logits_val), axis=-1)
                    logits_val = logits_val.detach().cpu()
                    #mic(logits)

                    g_logits.extend(logits_val)
                    g_labels.extend(x_labels_val)
                    g_masks.extend(x_mask_val)
                
                g_logits = [list(elem.numpy()) for elem in g_logits]
                g_labels = [list(elem.numpy()) for elem in g_labels]
                g_masks = [list(elem.numpy()) for elem in g_masks]

                predictions, labels = [], []
                predictions2, labels2 = [], []
                for pred, mask, label in zip(g_logits, g_masks, g_labels):
                    first_zero = mask.index(0)
                    predictions.append(pred[1:first_zero-1])
                    labels.append(label[1:first_zero-1])
                    predictions2.append(pred)
                    labels2.append(label)

        torch.save(model.state_dict(), os.path.join(Config.PATH_OUTPUT.value, "models", "bertneg_bioscope.model"))
        self.model = model

    def calculate_loss(self, attention_mask, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss()

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, 3)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        
        return loss_fct(active_logits, active_labels)  

    def convert_to_iob(values):
        int_to_iob = {0:'O',1:'I-ADR',2:'B-ADR'}
        return [[int_to_iob[value] for value in sample] for sample in values]

    def predict(self,df):
        df = df.sample(frac=1)
        tokenizer = Tokenizer(df=df,sample="tweet",span="correct_intervals",id="tweet_id", is_bioscope = False)
        tokenizer.tokenize_evaluation()
        df = tokenizer.get_augmented_df()
        bert_tokenizer = tokenizer.get_tokenizer()
        dataset = tokenizer.get_complete_dataset_evaluation()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        self.model.eval()
        g_logits, g_masks, g_tokens = [], [], []
        with torch.no_grad():
            for data in dataloader:
                # predictions
                x_ids, x_mask = data[0].to(self.device), data[1].to(self.device)
                tokens = tokenizer.get_tokens_from_ids(x_ids)
 
                logits = self.model(x_ids,x_mask)
                x_mask = x_mask.detach().cpu()
                x_mask.squeeze().tolist()

                logits = logits.detach().cpu()
                logits = torch.argmax(torch.FloatTensor(logits), axis=-1)
                logits = logits.detach().cpu()
                #mic(logits)

                g_logits.extend(logits)
                g_masks.extend(x_mask)
                g_tokens.extend(tokens)

            g_logits = [list(elem.numpy()) for elem in g_logits]
            g_masks = [list(elem.numpy()) for elem in g_masks]
            g_tokens = [list(elem) for elem in g_tokens]
            
            nlp = spacy.load("en_core_web_sm")
            # use mask
            new_df = pd.DataFrame({})
            for (idx,row), pred, mask, token in zip(df.iterrows(),g_logits, g_masks, g_tokens):
                first_zero = mask.index(0)
                tokens = []
                for t in row.bert_tokens:
                    if t not in bert_tokenizer.all_special_tokens:
                        tokens.append(t)
                intervals = get_predictions(row.tweet, tokens, pred[1:first_zero-1], bert_tokenizer, nlp)
                row['negation_intervals'] = intervals
                new_df = new_df.append(row)

            new_df = new_df.drop(columns=['bert_tokens', 'bert_ids','mask'])
            
            return new_df