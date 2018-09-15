import numpy as np
from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.annotations import load_marginals, load_gold_labels
from snorkel.learning.pytorch import LSTM
from load_external_annotations import load_external_labels
import json

with open('experiment.json', 'r') as jsonFile:
    experiment = json.load(jsonFile)
    jsonFile.close()

train_kwargs = experiment['parameters']['dis_model']

session = SnorkelSession()

ChemicalDisease = candidate_subclass('ChemicalDisease', ['chemical', 'disease'])

train = session.query(ChemicalDisease).filter(ChemicalDisease.split == 0).all()
dev = session.query(ChemicalDisease).filter(ChemicalDisease.split == 1).all()
test = session.query(ChemicalDisease).filter(ChemicalDisease.split == 2).all()

print('Training set:\t{0} candidates'.format(len(train)))
print('Dev set:\t{0} candidates'.format(len(dev)))
print('Test set:\t{0} candidates'.format(len(test)))

train_marginals = load_marginals(session, split=0)
L_gold_dev = load_gold_labels(session, annotator_name='gold', split=1)

lstm = LSTM(n_threads=None)
lstm.train(train, train_marginals, X_dev=dev, Y_dev=L_gold_dev, **train_kwargs)

load_external_labels(session, ChemicalDisease, split=2, annotator='gold')
L_gold_test = load_gold_labels(session, annotator_name='gold', split=2)

tp, fp, tn, fn = lstm.error_analysis(session, test, L_gold_test, set_unlabeled_as_neg=False)

metrics = {}
sent_lengths = set()
metrics[0] = [0 for _ in range(4)]

for i, metric in enumerate([tp, fp, tn, fn]):
    for sample in metric:
        metrics[0][i] += 1 
        cand_length = int(abs(sample.chemical.sentence.position - sample.disease.sentence.position)+1)
        if cand_length in sent_lengths:
            metrics[cand_length][i] += 1 
        else:
            metrics[cand_length] = [0 for _ in range(4)]
            metrics[cand_length][i] += 1 
            sent_lengths.add(cand_length)

experiment['results']['dis_model'] = metrics
     
with open("experiment.json", "w+") as jsonFile:
    json.dump(experiment, jsonFile)
    jsonFile.close()