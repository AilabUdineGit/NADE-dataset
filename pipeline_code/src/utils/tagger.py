#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.interval_merger import IntervalMerger
from spacy.training import  offsets_to_biluo_tags as biluo_tags_from_offsets
from spacy.lang.en import English
import numpy as np
import spacy
import re 

class Tag(object):


    def __init__(self, text_id, tokens, biluo, iob = None, io = None):
        self.text_id = text_id
        self.tokens = tokens
        self.biluo = biluo
        self.iob = iob
        self.io = io
        

    def __eq__(self, other):
        return self.text_id == other.text_id and \
               self.tokens == other.tokens and \
               self.biluo == other.biluo and \
               self.iob == other.iob and \
               self.io == other.io 



class Tagger(object):


    def __init__(self, text, annotations, pickle):
        self.text = text 
        self.annotations = annotations
        spacy_model = 'english'
        if spacy_model == 'english':
            self.nlp = English()

        self.text = self.tagging(self.text, self.annotations)

    def get(self):
        return self.text

    def tagging_custom(self, text, annotations, field_name):
        tags = []
        for i, t in text.iterrows():
            a = annotations[['begin','end',field_name]].loc[annotations['text_id']==t['text_id']]
            (tokens, biluo_tags) = self.biluo_tagging(t['text'], [tuple([int(x[0]), int(x[1]), str(x[2])]) for x in a.to_numpy()])
            tag = Tag(i, tokens, biluo_tags.copy())
            # tag.iob = self.biluo_to_iob(biluo_tags.copy())
            tag.io = self.biluo_to_io(biluo_tags.copy())
            tags.append(tag)

        text[f'{field_name}_tags'] = np.nan
        for t in tags:
            text[f'{field_name}_tags'].iloc[t.text_id] = [t if "-" not in t else t[2:] for t in t.io]
        
        return text

    def tagging(self, text, annotations):
        tags = []
        for i, t in text.iterrows():
            a = annotations[['begin','end','type']].loc[annotations['text_id']==t['text_id']]
            (tokens, biluo_tags) = self.biluo_tagging(t['text'], [tuple(x) for x in a.to_numpy()])
            tag = Tag(i, tokens, biluo_tags.copy())
            tag.iob = self.biluo_to_iob(biluo_tags.copy())
            tag.io = self.biluo_to_io(biluo_tags.copy())
            tags.append(tag)

        text['tokens'] = text['biluo_tags'] = text['io_tags'] = text['iob_tags'] = np.nan
        for t in tags:
            text['tokens'].iloc[t.text_id] = t.tokens
            text['biluo_tags'].iloc[t.text_id] = t.biluo
            text['iob_tags'].iloc[t.text_id] = t.iob
            text['io_tags'].iloc[t.text_id] = t.io
        
        if "polarity" in annotations.columns:
            text = self.tagging_custom(text, annotations, "polarity")
        if "modality" in annotations.columns:
            text = self.tagging_custom(text, annotations, "modality")
        
        return text


    def trim_annotations(self, annotations, text):
        tidy_annotations = []
        annotations = [(int(a),int(b),c) for (a,b,c) in annotations]
        for i, (b, e, a) in enumerate(annotations):
            if len(text) > b:
                while text[b] == ' ':
                    b += 1
                if len(text) >= e:
                    tidy_annotations.append((b, e, a))
        return tidy_annotations


    def tidy_text(self, text):
        for t in text.split():
            if self.is_special_token(t, '_'):
                new_t = t.replace("__", "--")
                text = text.replace(t, new_t)
        return text


    def biluo_tagging(self, text, annotations):
        annotations = self.trim_annotations(annotations, text)
        annotations = IntervalMerger().merge([[s,e] for (s,e,t) in annotations])
        annotations = [(s[0],s[1],"ADR") for s in annotations]
        text = self.tidy_text(text)
        doc = self.nlp(text)
        biluo_tags = biluo_tags_from_offsets(doc, annotations)
        tokens = [str(token) for token in doc]
        (tokens, biluo_tags) = self.postprocessing_0W30(tokens, biluo_tags, text, annotations)
        tokens = self.postprocessing_meaningless_tags(tokens)

        return (tokens, biluo_tags)


    def postprocessing_0W30(self, tokens, tags, text, annotations):
        b = 0
        while b < len(tags):
            t = tags[b]
            if t == '-':    # W030
                e = b + 1
                while len(tags) > e and tags[e] == '-':
                    e += 1
                e -= 1
                (token_chunks, tag_chunks) = self.split_tokens(b, e, tokens, tags, text, annotations)
                tokens = self.random_replace(tokens, b, e, token_chunks)
                tags = self.random_replace(tags, b, e, tag_chunks)
            b += 1
        return (tokens, tags)


    def postprocessing_meaningless_tags(self, tokens):
        for i, t in enumerate(tokens):
            if self.is_number(t):
                tokens[i] = 'number'
            elif self.is_hashtag(t):
                tokens[i] = t[1:len(t)]
            elif self.is_username(t):
                tokens[i] = 'username'
            elif self.is_link(t):
                tokens[i] = 'link'
            elif self.is_special_token(t, '-'):
                tokens[i] = t[2:-2]
            else:
                tokens[i] = self.delete_triples(t)
        return tokens


    def is_number(self, token):
        return token.replace('.','',1).replace(',','',1).isdigit()


    def is_hashtag(self, token):
        if len(token) > 0:
            return token[0] == '#' and len(token) > 1
        else:
            return False

    def is_username(self, token):
        if len(token) > 0:
            return token[0] == '@' and len(token) > 1
        else:
            return False


    def is_link(self, token):
        if len(token) > 4:
            return token[0:4] == 'http' or token[0:3] == 'www' 
        else:
            return False


    def is_special_token(self, token, separator = '_'):
        return True if token[0:2] == separator * 2 and token[-2:] == separator * 2 else False


    def delete_triples(self, token):
        new_token = ''
        for i in range(0, len(token) - 2):
            if not (token[i] == token[i+1] and token[i+1] == token[i+2]):
                new_token += token[i]            
        return new_token + token[len(token) - 2: len(token)]


    def random_replace(self, array, b, e, values):
        new_array = []
        for i, a in enumerate(array):
            if i == b:
                for v in values:
                    new_array.append(v)
            elif i > e or i < b: 
                new_array.append(a)
        return new_array


    def find_token(self, text, tokens, i):
        text_occ = [m.start() for m in re.finditer(re.escape(tokens[i]), text)]
        expected_occ = sum([len(t) + 1 for t in tokens[0:i]])
        occ = text_occ[0]
        dist = abs(text_occ[0] - expected_occ) 
        for t in text_occ:
            if abs(t - expected_occ) <= dist and \
                tokens[max(i-1, 0)] in text[max(0, t - (10 + len(tokens[i-1]))) : t] and \
                tokens[min(i+1, len(tokens) - 1)] in text[t : min(len(text), t + (10 + len(tokens[min(i+1, len(tokens)-1)])))]: 
                dist = abs(t - expected_occ) 
                occ = t 
        return occ


    def get_delta(self, span, token):
        for delta in range(0, len(token) + 1):
            if token[0 : len(token) - delta] in span:
                return delta


    def find_token_by_end_index(self, text, tokens, i, b, e, dist):
        min_dist = len(text)
        res = e
        res_delta = 0
        for j, t in enumerate(tokens[b:e]):
            delta = self.get_delta(text[i-len(t):i], t)
            if delta < len(t) and \
               abs((i - len(t)) - dist) < min_dist:
               res = j
               res_delta = delta
            dist += len(t) + 1
        return (res_delta, res)


    def find_annotations(self, annotations, b, e):
        a = []
        for (ab, ae, at) in annotations:
            if ab >= b and ae <= e: 
                a.append((ab, ae, at))       
        return a    


    def split_tokens(self, b, e, tokens, tags, text, annotations):
        token_chunks = [] 
        tag_chunks = []   
        b_token = self.find_token(text, tokens, b)
        e_token = self.find_token(text, tokens, e) + len(tokens[e])
        annotations = self.find_annotations(annotations, b_token, e_token)

        b_chunk = b 
        old_delta = 0
        old_e_chunk = 0
        for (ab, ae, at) in annotations:
            (new_delta, e_chunk) = self.find_token_by_end_index(text, tokens, ae, b, e, b_token)
            e_token = self.find_token(text, tokens, e_chunk) + len(tokens[e_chunk])
            (tmp_token_chunks, tmp_tag_chunks) = self.split_token(b_chunk, e_chunk, tokens, 
                                                                  (ab, ae, at), b_token, e_token)
            if old_delta == 0:
                b_chunk = e_chunk + 1
                token_chunks.extend(tmp_token_chunks)
                tag_chunks.extend(tmp_tag_chunks)
            elif e_chunk == old_e_chunk:
                b_chunk = e_chunk
                if len(token_chunks) > 0 and len(tag_chunks) > 0:
                    token_chunks.pop()
                    tag_chunks.pop()
                token_chunks.extend(tmp_token_chunks)
                tag_chunks.extend(tmp_tag_chunks)
            else:
                b_chunk = e_chunk + 1
                token_chunks.extend(tmp_token_chunks)
                tag_chunks.extend(tmp_tag_chunks)
            if b_chunk < len(tokens):
                b_token = self.find_token(text, tokens, b_chunk)
            old_delta = new_delta
            old_e_chunk = e_chunk
        return (token_chunks, tag_chunks)
        

    def split_token(self, b, e, tokens, annotation, b_token, e_token):
        token_chunks = [] 
        tag_chunks = [] 
        (ab, ae, at) = annotation
        if b == e: # U
            if b_token < ab and e_token == ae:
                token_chunks.extend([tokens[b][0 : ab - b_token], 
                                     tokens[b][ab - b_token : len(tokens[b])]])
                tag_chunks.extend(['O', 'U-' + at])
            elif b_token == ab and e_token > ae:
                token_chunks.extend([tokens[b][0 : len(tokens[e]) - (e_token - ae)], 
                                     tokens[b][len(tokens[e]) - (e_token - ae) : len(tokens[b])]])
                tag_chunks.extend(['U-'+at, 'O'])
            else:
                token_chunks.extend([tokens[b][0 : ab - b_token], 
                                     tokens[b][ab - b_token : len(tokens[e]) - (e_token - ae)],
                                     tokens[b][len(tokens[e]) - (e_token - ae) : len(tokens[b])]])
                tag_chunks.extend(['O', 'U-'+at, 'O'])
        else: # B I ... I L
            if ab == b_token:
                token_chunks.append(tokens[b])
                tag_chunks.append('B-' + at)
            else:
                token_chunks.extend([tokens[b][0 : ab - b_token], 
                                     tokens[b][ab - b_token : len(tokens[b])]])
                tag_chunks.extend(['O', 'B-' + at])
            for i in range(b + 1, e):
                token_chunks.append(tokens[i])
                tag_chunks.append('I-' + at)
            if ae == e_token:
                token_chunks.append(tokens[e])
                tag_chunks.append('L-' + at)
            else:
                token_chunks.extend([tokens[e][0 : len(tokens[e]) - (e_token - ae)], 
                                     tokens[e][len(tokens[e]) - (e_token - ae) : len(tokens[e])]])
                tag_chunks.extend(['L-'+at, 'O'])
        return (token_chunks, tag_chunks)


    def biluo_to_iob(self, biluo):
        for i, t in enumerate(biluo):
            if t[0:1] == 'U':
                biluo[i] = t.replace('U', 'B', 1)
            elif t[0:1] == 'L':
                biluo[i] = t.replace('L', 'I', 1)
        return biluo 


    def biluo_to_io(self, biluo):
        for i, t in enumerate(biluo):
            if t[0:1] == 'U':
                biluo[i] = t.replace('U', 'I', 1)
            elif t[0:1] == 'L':
                biluo[i] = t.replace('L', 'I', 1)
            elif t[0:1] == 'B':
                biluo[i] = t.replace('B', 'I', 1)
        return biluo 