def detokenize(self, sentence, labels, predictions):

    detok_sent = []
    detok_labels = []
    detok_predict = []

    for token, lab, pred in zip(sentence, labels, predictions):

        # CASE token to be added to the previous token
        if '##' in token:

            # rebuild the word
            detok_sent[-1] = detok_sent[-1] + token[2:]

            if pred > detok_predict[-1]:
                detok_predict[-1] = pred
            #    LOG.info(' > Prediction updated')

        else:
            detok_sent.append(token)
            detok_labels.append(lab)
            detok_predict.append(pred)

    return detok_sent, detok_labels, detok_predict