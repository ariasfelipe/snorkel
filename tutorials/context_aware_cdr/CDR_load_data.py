from snorkel import SnorkelSession
import os
from snorkel.parser import XMLMultiDocPreprocessor, CorpusParser
from utils import TaggerOneTagger
from snorkel.models import Document, Sentence, Candidate, candidate_subclass
from six.moves.cPickle import load
from snorkel.candidates import CrossContextPretaggedCandidateExtractor
import json 

with open('experiment.json', 'r') as jsonFile:
    experiment = json.load(jsonFile)
    jsonFile.close()

WINDOW_SIZE = experiment["parameters"]["window_size"]
if experiment["parameters"]["cand_lengths"] is False:
    CAND_LENGTHS = None
else:
    CAND_LENGTHS = experiment["parameters"]["cand_lengths"]

session = SnorkelSession()

# The following line is for testing only. Feel free to ignore it.
file_path = 'data/CDR.BioC.small.xml' if 'CI' in os.environ else 'data/CDR.BioC.xml'

doc_preprocessor = XMLMultiDocPreprocessor(
    path=file_path,
    doc='.//document',
    text='.//passage/text/text()',
    id='.//id/text()'
)

print('#'*20)
print('Parsing Documents')
print('#'*20)
tagger_one = TaggerOneTagger()
corpus_parser = CorpusParser(fn=tagger_one.tag)
corpus_parser.apply(list(doc_preprocessor))


experiment["stats"]["num_docs"] = session.query(Document).count()
experiment["stats"]["num_sents"] = session.query(Sentence).count()

ChemicalDisease = candidate_subclass('ChemicalDisease', ['chemical', 'disease'])

with open('data/doc_ids.pkl', 'rb') as f:
    train_ids, dev_ids, test_ids = load(f)

train_ids, dev_ids, test_ids = set(train_ids), set(dev_ids), set(test_ids)

train_sents, dev_sents, test_sents = [], [], []
docs = session.query(Document).order_by(Document.name).all()
for i, doc in enumerate(docs):
    if doc.name in train_ids:
        train_sents.append(list(doc.sentences))
    elif doc.name in dev_ids:
        dev_sents.append(list(doc.sentences))
    elif doc.name in test_ids:
        test_sents.append(list(doc.sentences))
    else:
        raise Exception('ID <{0}> not found in any id set'.format(doc.name))

print('#'*20)
print('Generating candidates')
print('#'*20)
candidate_extractor = CrossContextPretaggedCandidateExtractor(ChemicalDisease, ['Chemical', 'Disease'])
num_cands = []
for k, sents in enumerate([train_sents, dev_sents, test_sents]):
    candidate_extractor.apply(sents, window_size = WINDOW_SIZE, cand_lengths = CAND_LENGTHS, split=k)
    num_cands.append(session.query(ChemicalDisease).filter(ChemicalDisease.split == k).count())

experiment["stats"]["train_cands"] = num_cands[0]
experiment["stats"]["dev_cands"] = num_cands[1]
experiment["stats"]["test_cands"] = num_cands[2]

experiment['results']['gen_model'] = {}
experiment['results']['dis_model'] = {}

with open("experiment.json", "w") as jsonFile:
    json.dump(experiment, jsonFile)
    jsonFile.close()
