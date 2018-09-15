from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
import re
from snorkel.lf_helpers import (
    cross_context_get_tagged_text,
    cross_context_rule_regex_search_tagged_text,
    cross_context_rule_regex_search_btw_AB,
    cross_context_rule_regex_search_btw_BA,
    cross_context_rule_regex_search_before_A,
    cross_context_rule_regex_search_before_B,
    get_sentences,
    get_dep_path
)
from snorkel.annotations import LabelAnnotator, save_marginals, load_gold_labels
from load_external_annotations import load_external_labels
from snorkel.learning.structure import DependencySelector
from snorkel.learning import GenerativeModel
import bz2
from six.moves.cPickle import load
import matplotlib.pyplot as plt
import json

with open('experiment.json', 'r') as jsonFile:
    experiment = json.load(jsonFile)
    jsonFile.close()

#1 to evaluate on dev set, 2 to evaluate on test set
EVAL_SPLIT = experiment['parameters']['gen_model']['eval_split']
CALC_DEPS = experiment['parameters']['gen_model']['calc_deps']
LF_PROPENSITY = experiment['parameters']['gen_model']['lf_propensity']
DEP_THRESH = experiment['parameters']['gen_model']['dep_thresh']
DECAY =  experiment['parameters']['gen_model']['decay']
STEP_SIZE_CONST =  experiment['parameters']['gen_model']['step_size_const']
REG_PARAM = experiment['parameters']['gen_model']['reg_param']

session = SnorkelSession()

ChemicalDisease = candidate_subclass('ChemicalDisease', ['chemical', 'disease'])

train_cands = session.query(ChemicalDisease).filter(ChemicalDisease.split == 0).all()
dev_cands = session.query(ChemicalDisease).filter(ChemicalDisease.split == 1).all()
test_cands = session.query(ChemicalDisease).filter(ChemicalDisease.split == 2).all()

with bz2.BZ2File('data/ctd.pkl.bz2', 'rb') as ctd_f:
    ctd_unspecified, ctd_therapy, ctd_marker = load(ctd_f)

def cand_in_ctd_unspecified(c):
    return 1 if c.get_cids() in ctd_unspecified else 0

def cand_in_ctd_therapy(c):
    return 1 if c.get_cids() in ctd_therapy else 0

def cand_in_ctd_marker(c):
    return 1 if c.get_cids() in ctd_marker else 0

def LF_in_ctd_unspecified(c):
    return -1 * cand_in_ctd_unspecified(c)

def LF_in_ctd_therapy(c):
    return -1 * cand_in_ctd_therapy(c)

def LF_in_ctd_marker(c):
    return cand_in_ctd_marker(c)

# List to parenthetical
def ltp(x):
    return '(' + '|'.join(x) + ')'

def LF_induce(c):
    return 1 if re.search(r'{{A}}.{0,20}induc.{0,20}{{B}}', cross_context_get_tagged_text(c, session), flags=re.I) else 0

causal_past = ['induced', 'caused', 'due']
def LF_d_induced_by_c(c):
    return cross_context_rule_regex_search_btw_BA(c, '.{0,50}' + ltp(causal_past) + '.{0,9}(by|to).{0,50}', 1, session)
def LF_d_induced_by_c_tight(c):
    return cross_context_rule_regex_search_btw_BA(c, '.{0,50}' + ltp(causal_past) + ' (by|to) ', 1, session)

def LF_induce_name(c):
    return 1 if 'induc' in c.chemical.get_span().lower() else 0     

causal = ['cause[sd]?', 'induce[sd]?', 'associated with']
def LF_c_cause_d(c):
    return 1 if (
        re.search(r'{{A}}.{0,50} ' + ltp(causal) + '.{0,50}{{B}}', cross_context_get_tagged_text(c, session), re.I)
        and not re.search('{{A}}.{0,50}(not|no).{0,20}' + ltp(causal) + '.{0,50}{{B}}', cross_context_get_tagged_text(c, session), re.I)
    ) else 0

treat = ['treat', 'effective', 'prevent', 'resistant', 'slow', 'promise', 'therap']

def LF_d_treat_c(c):
    return cross_context_rule_regex_search_btw_BA(c, '.{0,50}' + ltp(treat) + '.{0,50}', -1, session)
def LF_c_treat_d(c):
    return cross_context_rule_regex_search_btw_AB(c, '.{0,50}' + ltp(treat) + '.{0,50}', -1, session)
def LF_treat_d(c):
    return cross_context_rule_regex_search_before_B(c, ltp(treat) + '.{0,50}', -1, session)
def LF_c_treat_d_wide(c):
    return cross_context_rule_regex_search_btw_AB(c, '.{0,200}' + ltp(treat) + '.{0,200}', -1, session)

def LF_c_d(c):
    return 1 if ('{{A}} {{B}}' in cross_context_get_tagged_text(c, session)) else 0

def LF_c_induced_d(c):
    return 1 if (
        ('{{A}} {{B}}' in cross_context_get_tagged_text(c, session)) and 
        (('-induc' in c[0].get_span().lower()) or ('-assoc' in c[0].get_span().lower()))
        ) else 0

def LF_improve_before_disease(c):
    return cross_context_rule_regex_search_before_B(c, 'improv.*', -1, session)

pat_terms = ['in a patient with ', 'in patients with']
def LF_in_patient_with(c):
    return -1 if re.search(ltp(pat_terms) + '{{B}}', cross_context_get_tagged_text(c, session), flags=re.I) else 0

uncertain = ['combin', 'possible', 'unlikely']
def LF_uncertain(c):
    return cross_context_rule_regex_search_before_A(c, ltp(uncertain) + '.*', -1, session)

def LF_induced_other(c):
    return cross_context_rule_regex_search_tagged_text(c, '{{A}}.{20,1000}-induced {{B}}', -1, session)

def LF_far_c_d(c):
    return cross_context_rule_regex_search_btw_AB(c, '.{100,5000}', -1, session)

def LF_far_d_c(c):
    return cross_context_rule_regex_search_btw_BA(c, '.{100,5000}', -1, session)

def LF_risk_d(c):
    return cross_context_rule_regex_search_before_B(c, 'risk of ', 1, session)

def LF_develop_d_following_c(c):
    return 1 if re.search(r'develop.{0,25}{{B}}.{0,25}following.{0,25}{{A}}', cross_context_get_tagged_text(c, session), flags=re.I) else 0

procedure, following = ['inject', 'administrat'], ['following']
def LF_d_following_c(c):
    return 1 if re.search('{{B}}.{0,50}' + ltp(following) + '.{0,20}{{A}}.{0,50}' + ltp(procedure), cross_context_get_tagged_text(c, session), flags=re.I) else 0

def LF_measure(c):
    return -1 if re.search('measur.{0,75}{{A}}', cross_context_get_tagged_text(c, session), flags=re.I) else 0

def LF_level(c):
    return -1 if re.search('{{A}}.{0,25} level', cross_context_get_tagged_text(c, session), flags=re.I) else 0

def LF_neg_d(c):
    return -1 if re.search('(none|not|no) .{0,25}{{B}}', cross_context_get_tagged_text(c, session), flags=re.I) else 0

WEAK_PHRASES = ['none', 'although', 'was carried out', 'was conducted',
                'seems', 'suggests', 'risk', 'implicated',
               'the aim', 'to (investigate|assess|study)']

uncertain = ['combin', 'possible', 'unlikely']


WEAK_RGX = r'|'.join(WEAK_PHRASES)

def LF_weak_assertions(c):
    return -1 if re.search(WEAK_RGX, cross_context_get_tagged_text(c, session), flags=re.I) else 0

def LF_ctd_marker_c_d(c):
    return LF_c_d(c) * cand_in_ctd_marker(c)

def LF_ctd_marker_induce(c):
    return (LF_c_induced_d(c) or LF_d_induced_by_c_tight(c)) * cand_in_ctd_marker(c)

def LF_ctd_therapy_treat(c):
    return LF_c_treat_d_wide(c) * cand_in_ctd_therapy(c)

def LF_ctd_unspecified_treat(c):
    return LF_c_treat_d_wide(c) * cand_in_ctd_unspecified(c)

def LF_ctd_unspecified_induce(c):
    return (LF_c_induced_d(c) or LF_d_induced_by_c_tight(c)) * cand_in_ctd_unspecified(c)


neg_keywords = set(['none', 'although', 'seems', 'suggests', 'risk', 'implicated',
               'aim', 'investigate', 'assess', 'study', 'possible', 'unlikely'])


# pos_keywords = set(['treat', 'effective', 'prevent', 'resistant', 'slow', 'promise', 'therap',
#                   'cause', 'caused', 'induce', 'due', 'induced', 'associated', 'increased', 'reduced'
#                   'increase', 'reduce', 'decrease', 'decreased'])
causal_past = ['induced', 'caused', 'due']
causal = ['cause[sd]?', 'induce[sd]?', 'associated with']
treat = ['treat', 'effective', 'prevent', 'resistant', 'slow', 'promise', 'therap']

pos_keywords = set(['treat', 'effective', 'prevent', 'resistant', 'slow', 'promise', 'therap',
                  'cause', 'caused', 'induce', 'due', 'induced'])



def LF_pos_keywords_in_dep_path(c):

    # if c.chemical.sentence.position == c.disease.sentence.position:
    #     return 0

    path = get_dep_path(c, session)
    words = set([word for (word, dep_type) in path])

    if len(words.intersection(pos_keywords)) > 0:
        return 1
    else:
        return 0

def LF_neg_keywords_in_dep_path(c):
    path = get_dep_path(c, session)
    words = set([word for (word, dep_type) in path])

    if len(words.intersection(neg_keywords)) > 0:
        return -1
    else:
        return 0

def LF_keywords_in_dep_path(c):

    # if c.chemical.sentence.position == c.disease.sentence.position:
    #     return 0

    path = get_dep_path(c, session)
    words = set([word for (word, dep_type) in path])

    if len(words.intersection(neg_keywords)) > len(words.intersection(pos_keywords)):
        return -1
    elif len(words.intersection(neg_keywords)) < len(words.intersection(pos_keywords)):
        return 1
    else:
        return 0


def LF_closer_chem(c):

    if c.chemical.sentence.position == c.disease.sentence.position:
        chem_start, chem_end = c.chemical.get_word_start(), c.chemical.get_word_end()
        dis_start, dis_end = c.disease.get_word_start(), c.disease.get_word_end()   
    # else:
    #     return 0 

    elif c.chemical.sentence.position < c.disease.sentence.position:
        chem_start, chem_end = c.chemical.get_word_start(), c.chemical.get_word_end()
        dis_start, dis_end = c.disease.get_word_start()+len(c.chemical.sentence.words), c.disease.get_word_end()+len(c.chemical.sentence.words)         
    else:
        chem_start, chem_end = c.chemical.get_word_start()+len(c.disease.sentence.words), c.chemical.get_word_end()+len(c.disease.sentence.words)
        dis_start, dis_end = c.disease.get_word_start(), c.disease.get_word_end()             

    if dis_start < chem_start:
        dist = chem_start - dis_end
    else:
        dist = dis_start - chem_end

    sentence_lists = [(sent.entity_types, sent.entity_cids) for sent in get_sentences(c, session)]
    entity_types, entity_cids = [], []

    for (entity_type_list, entity_cid_list) in sentence_lists:
        entity_types.extend(entity_type_list)
        entity_cids.extend(entity_cid_list)

    closest_other_chem = float('inf')

    for i in range(dis_end, min(len(entity_cids), dis_end + dist // 2)):
        et, cid = entity_types[i], entity_cids[i]
        if et == 'Chemical' and cid != entity_cids[chem_start]:
            return -1
    for i in range(max(0, dis_start - dist // 2), dis_start):
        et, cid = entity_types[i], entity_cids[i]
        if et == 'Chemical' and cid != entity_cids[chem_start]:
            return -1
    return 0
        
def LF_closer_dis(c):

    if c.chemical.sentence.position == c.disease.sentence.position:
        chem_start, chem_end = c.chemical.get_word_start(), c.chemical.get_word_end()
        dis_start, dis_end = c.disease.get_word_start(), c.disease.get_word_end()

    # else:
    #     return 0 

    elif c.chemical.sentence.position < c.disease.sentence.position:
        chem_start, chem_end = c.chemical.get_word_start(), c.chemical.get_word_end()
        dis_start, dis_end = c.disease.get_word_start()+len(c.chemical.sentence.words), c.disease.get_word_end()+len(c.chemical.sentence.words)         
    else:
        chem_start, chem_end = c.chemical.get_word_start()+len(c.disease.sentence.words), c.chemical.get_word_end()+len(c.disease.sentence.words)
        dis_start, dis_end = c.disease.get_word_start(), c.disease.get_word_end()             

    if dis_start < chem_start:
        dist = chem_start - dis_end
    else:
        dist = dis_start - chem_end
        
    # Try to find chemical disease than @dist/8 in either direction
    sentence_lists = [(sent.entity_types, sent.entity_cids) for sent in get_sentences(c, session)]
    entity_types, entity_cids = [], []

    for (entity_type_list, entity_cid_list) in sentence_lists:
        entity_types.extend(entity_type_list)
        entity_cids.extend(entity_cid_list)

    closest_other_chem = float('inf')
    
    for i in range(chem_end, min(len(entity_cids), chem_end + dist // 8)):
        et, cid = entity_types[i], entity_cids[i]
        if et == 'Disease' and cid != entity_cids[dis_start]:
            return -1
    for i in range(max(0, chem_start - dist // 8), chem_start):
        et, cid = entity_types[i], entity_cids[i]
        if et == 'Disease' and cid != entity_cids[dis_start]:
            return -1
    return 0

 

LFs = [
    # LF_closer_dis,
    # LF_closer_chem,
    #LF_keywords_in_dep_path,
    LF_pos_keywords_in_dep_path,   
    #LF_neg_keywords_in_dep_path,
    LF_c_cause_d,
    LF_c_d,
    LF_c_induced_d,
    LF_c_treat_d,
    LF_c_treat_d_wide,
    LF_ctd_marker_c_d,
    LF_ctd_marker_induce,
    LF_ctd_therapy_treat,
    LF_ctd_unspecified_treat,
    LF_ctd_unspecified_induce,
    LF_d_following_c,
    LF_d_induced_by_c,
    LF_d_induced_by_c_tight,
    LF_d_treat_c,
    LF_develop_d_following_c,
    LF_far_c_d,
    LF_far_d_c,
    LF_improve_before_disease,
    LF_in_ctd_therapy,
    LF_in_ctd_marker,
    LF_in_patient_with,
    LF_induce,
    LF_induce_name,
    LF_induced_other,
    LF_level,
    LF_measure,
    LF_neg_d,
    LF_risk_d,
    LF_treat_d,
    LF_uncertain,
    LF_weak_assertions,
]

experiment['parameters']['lfs'] = [lf.__name__ for lf in LFs]

status = 'Creating Label Matrix'
print('#'*len(status))
print(status)
print('#'*len(status))

labeler = LabelAnnotator(lfs=LFs)
L_train = labeler.apply(split=0)

lfs_train_data_stats = L_train.lf_stats(session)
lfs_train_data_stats.to_csv('stats/lfs_train_data_stats.csv')

if CALC_DEPS:
    ds = DependencySelector()
    deps = ds.select(L_train, threshold=DEP_THRESH)
else:
    deps = ()

status = 'Training Generative Model'
print('#'*len(status))
print(status)
print('#'*len(status))


gen_model = GenerativeModel(lf_propensity=LF_PROPENSITY)
gen_model.train(
    L_train, deps=deps, decay=DECAY, step_size=STEP_SIZE_CONST/L_train.shape[0], reg_param=REG_PARAM
)

train_marginals = gen_model.marginals(L_train)
plt.hist(train_marginals, bins=20)
plt.savefig('stats/marginals_distribution.png')

lfs_learned_train_data_stats = gen_model.learned_lf_stats()
lfs_learned_train_data_stats.index = lfs_train_data_stats.index
lfs_learned_train_data_stats.to_csv('stats/lfs_learned_train_data_stats.csv')

save_marginals(session, L_train, train_marginals)

status = 'Labeling Evaluation data'
print('#'*len(status))
print(status)
print('#'*len(status))

load_external_labels(session, ChemicalDisease, split=EVAL_SPLIT, annotator='gold')
L_gold = load_gold_labels(session, annotator_name='gold', split=EVAL_SPLIT)
L = labeler.apply_existing(split=EVAL_SPLIT)

tp, fp, tn, fn = gen_model.error_analysis(session, L, L_gold, set_unlabeled_as_neg=False)

# positives = []
# positives.extend(tp)

# pos_dic = {}
# for sample in positives:
#     path = get_dep_path(sample, session)
#     for n, _ in path:
#         if n not in pos_dic:
#             pos_dic[n] = 1
#         else:
#             pos_dic[n]+=1

# pos_sorted_keys = sorted(pos_dic, key=pos_dic.get, reverse=True)
# for r in pos_sorted_keys[:50]:
#     print(r, pos_dic[r])



# negatives = []
# negatives.extend(fp)

# print('#'*len(status))

# neg_dic = {}
# for sample in negatives:
#     path = get_dep_path(sample, session)
#     for n, _ in path:
#         if n not in neg_dic:
#             neg_dic[n] = 1
#         else:
#             neg_dic[n]+=1

# neg_sorted_keys = sorted(neg_dic, key=neg_dic.get, reverse=True)
# for r in neg_sorted_keys[:50]:
#     print(r, neg_dic[r])

lf_eval_stats = L.lf_stats(session, L_gold, gen_model.learned_lf_stats()['Accuracy'])
lf_eval_stats.to_csv('stats/lf_eval_stats.csv')

#Create a dictionary for the tp, fp, tn, fn count for every length of candidates (0 is for the combined counts)
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
        
experiment['results']['gen_model'] = metrics
experiment['results']['dis_model'] = {}

with open("experiment.json", "w+") as jsonFile:
    json.dump(experiment, jsonFile)
    jsonFile.close()
