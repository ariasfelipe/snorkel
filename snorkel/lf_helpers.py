from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import numpy as np
import re
from itertools import chain
from collections import deque


from snorkel.annotations import load_gold_labels
from snorkel.learning.utils import MentionScorer
from snorkel.models import Span, Label, Candidate, Sentence
from snorkel.utils import tokens_to_ngrams


def get_text_splits(c):
    """
    Given a k-arity Candidate defined over k Spans, return the chunked parent
    context (e.g. Sentence) split around the k constituent Spans.

    NOTE: Currently assumes that these Spans are in the same Context
    """
    spans = []
    for i, span in enumerate(c.get_contexts()):

        # Note: {{0}}, {{1}}, etc. does not work as an un-escaped regex pattern,
        # hence A, B, ...
        try:
            spans.append((span.char_start, span.char_end, chr(65+i)))
        except AttributeError:
            raise ValueError(
                "Only handles Contexts with char_start, char_end attributes.")
    spans.sort()

    # NOTE: Assume all Spans in same parent Context
    text = span.get_parent().text

    # Get text chunks
    chunks = [text[:spans[0][0]], "{{%s}}" % spans[0][2]]
    for j in range(len(spans)-1):
        chunks.append(text[spans[j][1]+1:spans[j+1][0]])
        chunks.append("{{%s}}" % spans[j+1][2])
    chunks.append(text[spans[-1][1]+1:])
    return chunks


def cross_context_get_text_splits(c, session):
    """
    Given a k-arity Candidate defined over k Spans, return the chunked parent
    context (e.g. Sentence) split around the k constituent Spans.
    """

    spans = []

    context_list = [(span.sentence.position, span) for span in c.get_contexts()]
    context_list.sort(key=lambda x: x[0])

    position_offset = context_list[0][0]

    char_offsets = []
    char_offsets.append(0)

    for i in range(len(context_list)-1):
        #Changed order of indices
        if context_list[i][0] != context_list[i+1][0]:
            char_offsets.append(context_list[i+1][1].sentence.abs_char_offsets[0] - context_list[i][1].sentence.abs_char_offsets[-1] + context_list[i][1].sentence.char_offsets[-1]) 

    for i, span in enumerate(c.get_contexts()):
        try:
            spans.append((span.char_start+char_offsets[span.sentence.position - position_offset], span.char_end+char_offsets[span.sentence.position - position_offset], chr(65+i)))

        except AttributeError:
            raise ValueError(
                "Only handles Contexts with char_start, char_end attributes.")
    spans.sort()

    sents = get_sentences(c, session)
    text_list = [sent.text for sent in sents]
    text = ' '.join(text_list)

    chunks = [text[:spans[0][0]], "{{%s}}" % spans[0][2]]
    for j in range(len(spans)-1):
        chunks.append(text[spans[j][1]+1:spans[j+1][0]])
        chunks.append("{{%s}}" % spans[j+1][2])
    chunks.append(text[spans[-1][1]+1:])

    return chunks


def get_tagged_text(c):
    """
    Returns the text of c's parent context with c's unary spans replaced with
    tags {{A}}, {{B}}, etc. A convenience method for writing LFs based on e.g.
    regexes.
    """
    return "".join(get_text_splits(c))

def cross_context_get_tagged_text(c, session):
    """
    Returns the text of c's parent context with c's unary spans replaced with
    tags {{A}}, {{B}}, etc. A convenience method for writing LFs based on e.g.
    regexes.
    """
    return "".join(cross_context_get_text_splits(c, session))


def get_text_between(c):
    """
    Returns the text between the two unary Spans of a binary-Span Candidate,
    where both are in the same Sentence.
    """
    chunks = get_text_splits(c)
    if len(chunks) == 5:
        return chunks[2]
    else:
        raise ValueError("Only applicable to binary Candidates")


def is_inverted(c):
    """Returns True if the ordering of the candidates in the sentence is
    inverted."""
    if len(c) != 2:
        raise ValueError("Only applicable to binary Candidates")
    return c[0].get_word_start() > c[1].get_word_start()


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


def cross_ctxt_get_between_tokens(c, session, attrib='words', n_max=1, case_sensitive=False):
    """Returns a list of generators (one per sentence in candidate) to all the tokens between the mentions"""
    if len(c) != 2:
        raise ValueError("Only applicable to binary Candidates")

    span0 = c[0]
    span1 = c[1]

    sentences = get_sentences(c, session, attrib, n_max, case_sensitive)
    num_sentences = len(sentences)

    if num_sentences == 1:
        return get_between_tokens(c, attrib, n_max, case_sensitive)
    else:
        f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
        if span0.sentence.position < span1.sentence.position:
            first_mention = span0
            second_mention = span1 
        else:
            first_mention = span1
            second_mention = span0

        generator_list = []

        f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
        generator_list.append(tokens_to_ngrams(list(map(f, sentences[0]._asdict()[attrib][first_mention.get_word_end():])), n_max=n_max))

        for i in range(1, num_sentences - 1):
            generator_list.append(tokens_to_ngrams(list(map(f, sentences[i]._asdict()[attrib][:])), n_max=n_max))

        generator_list.append(tokens_to_ngrams(list(map(f, sentences[-1]._asdict()[attrib][:second_mention.get_word_start()])), n_max=n_max))

        return chain(*generator_list)

def cross_ctxt_get_left_tokens(c, session, window=3, attrib='words', n_max=1, case_sensitive=False):
    """
    Return the tokens within a window to the _left_ of the Candidate.
    For higher-arity Candidates, defaults to the _first_ argument.
    :param window: The number of tokens to the left of the first argument to
        return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """

    first_span = None

    for cand in c:
        if first_span is None or cand.sentence.position < first_span.sentence.position:
            first_span = cand

    i = first_span.get_word_start()

    sentences = get_sentences(c, session, attrib, n_max, case_sensitive)

    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return tokens_to_ngrams(list(map(f,
        sentences[0]._asdict()[attrib][max(0, i-window):i])), n_max=n_max)

def shortest_path(graph, start, goal):
    queue = deque([(start, [start])])
    while queue:
        (vertex, path) = queue.popleft()
        for node in graph[vertex]['neighbors'] - set(path):
            if node == goal:
                chosen_path = path + [node]
                return [(graph[step]['word'], graph[step]['dep_type'])for step in chosen_path]
            else:
                queue.append((node, path + [node]))

def get_dep_path(c, session):

    span0 = c[0]
    span1 = c[1]
    
    if span0.sentence.position < span1.sentence.position:
        first_offset = span0.get_word_start()+1
        second_offset = span1.get_word_start()+len(span0.sentence._asdict()['words']) +1
    elif span0.sentence.position > span1.sentence.position:
        first_offset = span1.get_word_start()+1
        second_offset = span0.get_word_start()+len(span1.sentence._asdict()['words'])+1
    else:
        first_offset = span0.get_word_start()+1
        second_offset = span1.get_word_start()+1

    dep_heads, dep_labels, words = [], [], []
    sents = get_sentences(c, session)

    dep_heads.extend(sents[0]._asdict()['dep_parents'])
    past_root = dep_heads.index(0) + 1
    dep_labels.extend(sents[0]._asdict()['dep_labels'])
    words.extend(sents[0]._asdict()['words'])

    for i in range(1, len(sents)):

        temp_heads = []

        for j, head in enumerate(sents[i]._asdict()['dep_parents']):
            if head == 0:
                temp_heads.append(past_root)
                past_root = j + 1 + len(dep_heads)
            else:
                temp_heads.append(head+len(dep_heads))

        dep_heads.extend(temp_heads)
        dep_labels.extend(sents[i]._asdict()['dep_labels'])
        words.extend(sents[i]._asdict()['words'])


    dep_tree = {index: {'neighbors':set()} for index in range(1, len(dep_heads)+1)}

    for i, (parent, label, word) in enumerate(zip(dep_heads, dep_labels, words), 1):
        dep_tree[i]['dep_type'] = label
        dep_tree[i]['word'] = word
        if parent != 0:
            dep_tree[i]['neighbors'].add(parent)
        if parent != 0:
            dep_tree[parent]['neighbors'].add(i)
              
    return shortest_path(dep_tree, first_offset, second_offset)

def cross_ctxt_get_right_tokens(c, session, window=3, attrib='words', n_max=1,
    case_sensitive=False):
    """
    Return the tokens within a window to the _right_ of the Candidate.
    For higher-arity Candidates, defaults to the _last_ argument.
    :param window: The number of tokens to the right of the last argument to
        return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """

    last_span = None

    for cand in c:
        if last_span is None or cand.sentence.position > last_span.sentence.position:
            last_span = cand
    i = last_span.get_word_end()

    sentences = get_sentences(c, session, attrib, n_max, case_sensitive)

    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return tokens_to_ngrams(list(map(f,
        sentences[-1]._asdict()[attrib][i+1:i+1+window])), n_max=n_max)


def get_between_tokens(c, attrib='words', n_max=1, case_sensitive=False):
    """
    TODO: write doc_string
    """
    if len(c) != 2:
        raise ValueError("Only applicable to binary Candidates")
    span0 = c[0]
    span1 = c[1]
    if span0.get_word_start() < span1.get_word_start():
        left_span = span0
        dist_btwn = span1.get_word_start() - span0.get_word_end() - 1
    else:
        left_span = span1
        dist_btwn = span0.get_word_start() - span1.get_word_end() - 1
    return get_right_tokens(left_span, window=dist_btwn, attrib=attrib,
        n_max=n_max, case_sensitive=case_sensitive)


def get_left_tokens(c, window=3, attrib='words', n_max=1, case_sensitive=False):
    """
    Return the tokens within a window to the _left_ of the Candidate.
    For higher-arity Candidates, defaults to the _first_ argument.
    :param window: The number of tokens to the left of the first argument to
        return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    try:
        span = c
        i = span.get_word_start()
    except:
        span = c[0]
        i = span.get_word_start()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return tokens_to_ngrams(list(map(f,
        span.get_parent()._asdict()[attrib][max(0, i-window):i])), n_max=n_max)


def get_right_tokens(c, window=3, attrib='words', n_max=1,
    case_sensitive=False):
    """
    Return the tokens within a window to the _right_ of the Candidate.
    For higher-arity Candidates, defaults to the _last_ argument.
    :param window: The number of tokens to the right of the last argument to
        return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    try:
        span = c
        i = span.get_word_end()
    except:
        span = c[-1]
        i = span.get_word_end()
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return tokens_to_ngrams(list(map(f,
        span.get_parent()._asdict()[attrib][i+1:i+1+window])), n_max=n_max)


def contains_token(c, tok, attrib='words', case_sensitive=False):
    """
    Checks if any of the contituent Spans contain a token
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    try:
        spans = c.get_contexts()
    except:
        spans = [c]
    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return f(tok) in set(chain.from_iterable(map(f, span.get_attrib_tokens(attrib))
        for span in spans))


def get_doc_candidate_spans(c):
    """
    Get the Spans in the same document as Candidate c, where these Spans are
    arguments of Candidates.
    """
    # TODO: Fix this to be more efficient and properly general!!
    spans = list(chain.from_iterable(s.spans for s in c[0].get_parent().document.sentences))
    return [s for s in spans if s != c[0]]


def get_sent_candidate_spans(c):
    """
    Get the Spans in the same Sentence as Candidate c, where these Spans are
    arguments of Candidates.
    """
    # TODO: Fix this to be more efficient and properly general!!
    return [s for s in c[0].get_parent().spans if s != c[0]]


def get_matches(lf, candidate_set, match_values=[1,-1]):
    """
    A simple helper function to see how many matches (non-zero by default) an LF gets.
    Returns the matched set, which can then be directly put into the Viewer.
    """
    matches = []
    for c in candidate_set:
        label = lf(c)
        if label in match_values:
            matches.append(c)
    print("%s matches" % len(matches))
    return matches


def rule_text_btw(candidate, text, sign):
    return sign if text in get_text_between(candidate) else 0


def rule_text_in_span(candidate, text, span, sign):
    return sign if text in candidate[span].get_span().lower() else 0


def cross_context_rule_regex_search_tagged_text(candidate, pattern, sign, session):
    return sign if re.search(pattern, cross_context_get_tagged_text(candidate, session), flags=re.I) else 0


def rule_regex_search_btw_AB(candidate, pattern, sign):
    return sign if re.search(r'{{A}}' + pattern + r'{{B}}', get_tagged_text(candidate), flags=re.I) else 0


def cross_context_rule_regex_search_btw_AB(candidate, pattern, sign, session):
    return sign if re.search(r'{{A}}' + pattern + r'{{B}}', cross_context_get_tagged_text(candidate, session), flags=re.I) else 0


def rule_regex_search_btw_BA(candidate, pattern, sign):
    return sign if re.search(r'{{B}}' + pattern + r'{{A}}', get_tagged_text(candidate), flags=re.I) else 0


def cross_context_rule_regex_search_btw_BA(candidate, pattern, sign, session):
    return sign if re.search(r'{{B}}' + pattern + r'{{A}}', cross_context_get_tagged_text(candidate, session), flags=re.I) else 0


def rule_regex_search_before_A(candidate, pattern, sign):
    return sign if re.search(pattern + r'{{A}}.*{{B}}', get_tagged_text(candidate), flags=re.I) else 0


def cross_context_rule_regex_search_before_A(candidate, pattern, sign, session):
    return sign if re.search(pattern + r'{{A}}.*{{B}}', cross_context_get_tagged_text(candidate, session), flags=re.I) else 0


def rule_regex_search_before_B(candidate, pattern, sign):
    return sign if re.search(pattern + r'{{B}}.*{{A}}', get_tagged_text(candidate), flags=re.I) else 0


def cross_context_rule_regex_search_before_B(candidate, pattern, sign, session):
    return sign if re.search(pattern + r'{{B}}.*{{A}}', cross_context_get_tagged_text(candidate, session), flags=re.I) else 0


def test_LF(session, lf, split, annotator_name):
    """
    Gets the accuracy of a single LF on a split of the candidates, w.r.t. annotator labels,
    and also returns the error buckets of the candidates.
    """
    test_candidates = session.query(Candidate).filter(Candidate.split == split).all()
    test_labels     = load_gold_labels(session, annotator_name=annotator_name, split=split)
    scorer          = MentionScorer(test_candidates, test_labels)
    test_marginals  = np.array([0.5 * (lf(c) + 1) for c in test_candidates])
    return scorer.score(test_marginals, set_unlabeled_as_neg=False, set_at_thresh_as_neg=False)
