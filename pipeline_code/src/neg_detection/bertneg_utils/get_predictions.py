from icecream import ic

def get_max_pred(p1, p2):
    if "B" in [p1,p2]:
        return "B"
    elif "I" in [p1,p2]:
        return "I"
    else:
        return "O"

def token_to_original(original_text, tokens, tokenizer, nlp):
        
    spacy_tokens = [x for x in nlp(original_text)]
    
    token_2_original = {}
    tok_index = 0
    tokens_copy = tokens#.copy()

    special = tokenizer.all_special_tokens
    for stok in spacy_tokens:
        char_begin = stok.idx
        char_end = stok.idx+len(stok)

        stok_tokens = tokenizer.tokenize(stok.text)
        if len(stok_tokens)==0:
            continue

        acc_tok = []
        acc_index = []

        while stok_tokens != acc_tok:
            tok = tokens_copy.pop(0)
            if tok not in special:
                acc_tok.append(tok)
                acc_index.append(tok_index)
            if tok == tokenizer.unk_token:
                acc_tok = stok_tokens
                acc_index.append(tok_index)
            tok_index+=1

        for idx, tok in zip(acc_index, acc_tok):
            token_2_original[idx] = {
                "bert_token": tok,
                "original_text": stok.text,
                "char_begin": char_begin,
                "char_end": char_end,
            }
            
    return token_2_original

def get_predictions(original_text, tokens, preds, tokenizer, nlp):
    mapping = token_to_original(original_text, tokens, tokenizer, nlp)

    pred_ents_token_lvl = []

    tmp_ent = False
    i=0
    while i<len(preds):
        if preds[i] == 2 or preds[i]==1:
            start_ent = i
            i += 1
            while i<len(preds) and preds[i] == 1:
                i += 1
            end_ent = i
            pred_ents_token_lvl.append((start_ent,end_ent))
        else:
            i += 1

    pred_ents_char_lvl = []

    for start,end in pred_ents_token_lvl:
        if mapping.get(start) or mapping.get(end):
        
            pred_ents_char_lvl.append((
                mapping.get(start)["char_begin"],
                mapping.get(end-1)["char_end"],
            ))
    
    return pred_ents_char_lvl
