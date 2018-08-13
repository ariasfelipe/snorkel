from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from itertools import chain
from builtins import *
from future.utils import iteritems
from snorkel.utils import tokens_to_ngrams
from snorkel.models import Sentence



class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols"""
    def __init__(self, starting_symbol=2, unknown_symbol=1): 
        self.s       = starting_symbol
        self.unknown = unknown_symbol
        self.d       = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in iteritems(self.d)}


def scrub(s):
    return ''.join(c for c in s if ord(c) < 128)

#Global to enable  access database
session = None


def get_sentences(c, session, attrib='words', n_max=1, case_sensitive=False):
    """
    Returns an ordered list of all the sentences associated with the candidate
    """
    parents = c.get_unique_parents()
    if len(parents) == 1:
        return [parents[0]]   
    elif len(parents) == 2:
        return [parents[0], parents[1]] if parents[0].position < parents[1].position else [parents[1], parents[0]]
    else:
        doc_id = parents[0].document_id
        positions = [parent.position for parent in parents]
        first_idx = min(positions)
        last_idx = max(positions)
        return session.query(Sentence).filter(Sentence.document_id == doc_id).filter(Sentence.position >= first_idx).filter(Sentence.position <= last_idx).oder_by(Sentence.position).all()


def get_all_tokens(c, session, attrib='words', n_max=1, case_sensitive=False):
    """Returns a generator to all the tokens in a candidate (cross sentence enabled)"""
    sentences = get_sentences(c, session, attrib, n_max, case_sensitive)
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return chain(*[tokens_to_ngrams(list(map(f, sentence._asdict()[attrib][:])), n_max=n_max) for sentence in sentences])


def enable_session(snorkel_session):
    global session 
    if session is None:
        session = snorkel_session


def candidate_to_tokens(candidate, token_type='words'):
    if session is None and len(candidate.get_unique_parents()) > 2:
        raise ValueError('Session must be enabled with enable_session(snorkel_session) for cross context candidates')
    tokens = get_all_tokens(candidate, session, attrib=token_type)  
    return [scrub(w).lower() for w in tokens]

def create_candidate_args(candidate):

    context_list = [(span.sentence.position, span) for span in candidate.get_contexts()]
    context_list.sort(key=lambda x: x[0])

    position_offset = context_list[0][0]

    word_offsets = []
    word_offsets.append(0)

    word_count = 0
    for i in range(len(context_list)-1):
        if context_list[i][0] != context_list[i+1][0]:
            word_count += len(context_list[i][1].sentence.words)
            word_offsets.append(word_count) 

    args = [
        (candidate[0].get_word_start() + word_offsets[candidate[0].sentence.position - position_offset], candidate[0].get_word_end() + word_offsets[candidate[0].sentence.position - position_offset], 1),
        (candidate[1].get_word_start() + word_offsets[candidate[1].sentence.position - position_offset], candidate[1].get_word_end() + word_offsets[candidate[1].sentence.position - position_offset], 2)
    ]

    return args