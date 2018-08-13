from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from future.utils import iteritems

from collections import defaultdict, deque
from copy import deepcopy
from itertools import product
import re
from sqlalchemy.sql import select

from snorkel.models import Candidate, TemporarySpan, Sentence
from snorkel.udf import UDF, UDFRunner

QUEUE_COLLECT_TIMEOUT = 5


class CandidateExtractor(UDFRunner):
    """
    An operator to extract Candidate objects from a Context.

    :param candidate_class: The type of relation to extract, defined using
                            :func:`snorkel.models.candidate_subclass <snorkel.models.candidate.candidate_subclass>`
    :param cspaces: one or list of :class:`CandidateSpace` objects, one for each relation argument. Defines space of
                    Contexts to consider
    :param matchers: one or list of :class:`snorkel.matchers.Matcher` objects, one for each relation argument. Only tuples of
                     Contexts for which each element is accepted by the corresponding Matcher will be returned as Candidates
    :param self_relations: Boolean indicating whether to extract Candidates that relate the same context.
                           Only applies to binary relations. Default is False.
    :param nested_relations: Boolean indicating whether to extract Candidates that relate one Context with another
                             that contains it. Only applies to binary relations. Default is False.
    :param symmetric_relations: Boolean indicating whether to extract symmetric Candidates, i.e., rel(A,B) and rel(B,A),
                                where A and B are Contexts. Only applies to binary relations. Default is False.
    """
    def __init__(self, candidate_class, cspaces, matchers, self_relations=False, nested_relations=False, symmetric_relations=False):
        super(CandidateExtractor, self).__init__(CandidateExtractorUDF,
                                                 candidate_class=candidate_class,
                                                 cspaces=cspaces,
                                                 matchers=matchers,
                                                 self_relations=self_relations,
                                                 nested_relations=nested_relations,
                                                 symmetric_relations=symmetric_relations)

    def apply(self, xs, split=0, **kwargs):
        super(CandidateExtractor, self).apply(xs, split=split, **kwargs)

    def clear(self, session, split, **kwargs):
        session.query(Candidate).filter(Candidate.split == split).delete()


class CandidateExtractorUDF(UDF):
    def __init__(self, candidate_class, cspaces, matchers, self_relations, nested_relations, symmetric_relations, **kwargs):
        self.candidate_class     = candidate_class
        # Note: isinstance is the way to check types -- not type(x) in [...]!
        self.candidate_spaces    = cspaces if isinstance(cspaces, (list, tuple)) else [cspaces]
        self.matchers            = matchers if isinstance(matchers, (list, tuple)) else [matchers]
        self.nested_relations    = nested_relations
        self.self_relations      = self_relations
        self.symmetric_relations = symmetric_relations

        # Check that arity is same
        if len(self.candidate_spaces) != len(self.matchers):
            raise ValueError("Mismatched arity of candidate space and matcher.")
        else:
            self.arity = len(self.candidate_spaces)

        # Make sure the candidate spaces are different so generators aren't expended!
        self.candidate_spaces = list(map(deepcopy, self.candidate_spaces))

        # Preallocates internal data structures
        self.child_context_sets = [None] * self.arity
        for i in range(self.arity):
            self.child_context_sets[i] = set()

        super(CandidateExtractorUDF, self).__init__(**kwargs)

    def apply(self, context, clear, split, **kwargs):
        # Generate TemporaryContexts that are children of the context using the candidate_space and filtered
    	# by the Matcher
        for i in range(self.arity):
            self.child_context_sets[i].clear()
            for tc in self.matchers[i].apply(self.candidate_spaces[i].apply(context)):
                tc.load_id_or_insert(self.session)
                self.child_context_sets[i].add(tc)

        # Generates and persists candidates
        extracted = set()
        candidate_args = {'split': split}
        for args in product(*[enumerate(child_contexts) for child_contexts in self.child_context_sets]):

            # TODO: Make this work for higher-order relations
            if self.arity == 2:

                ai, a = args[0]
                bi, b = args[1]

                # Check for self-joins, "nested" joins (joins from span to its subspan), and flipped duplicate
                # "symmetric" relations. For symmetric relations, if mentions are of the same type, maintain
                # their order in the sentence.
                if not self.self_relations and a == b:
                    continue
                elif not self.nested_relations and (a in b or b in a):
                    continue
                elif not self.symmetric_relations and ((b, a) in extracted or
                    (self.matchers[0] == self.matchers[1] and a.char_start > b.char_start)):
                    continue

                # Keep track of extracted
                extracted.add((a,b))

            # Assemble candidate arguments
            for i, arg_name in enumerate(self.candidate_class.__argnames__):
            	candidate_args[arg_name + '_id'] = args[i][1].id

            # Checking for existence
            if not clear:
                q = select([self.candidate_class.id])
                for key, value in iteritems(candidate_args):
                    q = q.where(getattr(self.candidate_class, key) == value)
                candidate_id = self.session.execute(q).first()
                if candidate_id is not None:
                    continue

            # Add Candidate to session
            yield self.candidate_class(**candidate_args)
            

class CrossContextCandidateExtractor(UDFRunner):
    """
    An operator to extract Candidate objects from a Context.

    :param candidate_class: The type of relation to extract, defined using
                            :func:`snorkel.models.candidate_subclass <snorkel.models.candidate.candidate_subclass>`
    :param cspaces: one or list of :class:`CandidateSpace` objects, one for each relation argument. Defines space of
                    Contexts to consider
    :param matchers: one or list of :class:`snorkel.matchers.Matcher` objects, one for each relation argument. Only tuples of
                     Contexts for which each element is accepted by the corresponding Matcher will be returned as Candidates
    :param self_relations: Boolean indicating whether to extract Candidates that relate the same context.
                           Only applies to binary relations. Default is False.
    :param nested_relations: Boolean indicating whether to extract Candidates that relate one Context with another
                             that contains it. Only applies to binary relations. Default is False.
    :param symmetric_relations: Boolean indicating whether to extract symmetric Candidates, i.e., rel(A,B) and rel(B,A),
                                where A and B are Contexts. Only applies to binary relations. Default is False.
    """
    def __init__(self, candidate_class, cspaces, matchers, self_relations=False, nested_relations=False, symmetric_relations=False):
        super(CrossContextCandidateExtractor, self).__init__(CrossContextCandidateExtractorUDF,
                                                 candidate_class=candidate_class,
                                                 cspaces=cspaces,
                                                 matchers=matchers,
                                                 self_relations=self_relations,
                                                 nested_relations=nested_relations,
                                                 symmetric_relations=symmetric_relations)

    def apply(self, xs, split=0, **kwargs):
        super(CrossContextCandidateExtractor, self).apply(xs, split=split, **kwargs)

    def clear(self, session, split, **kwargs):
        session.query(Candidate).filter(Candidate.split == split).delete()


class CrossContextCandidateExtractorUDF(UDF):
    def __init__(self, candidate_class, cspaces, matchers, self_relations, nested_relations, symmetric_relations, **kwargs):
        self.candidate_class     = candidate_class
        # Note: isinstance is the way to check types -- not type(x) in [...]!
        self.candidate_spaces    = cspaces if isinstance(cspaces, (list, tuple)) else [cspaces]
        self.matchers            = matchers if isinstance(matchers, (list, tuple)) else [matchers]
        self.nested_relations    = nested_relations
        self.self_relations      = self_relations
        self.symmetric_relations = symmetric_relations

        # Check that arity is same
        if len(self.candidate_spaces) != len(self.matchers):
            raise ValueError("Mismatched arity of candidate space and matcher.")
        else:
            self.arity = len(self.candidate_spaces)

        # Make sure the candidate spaces are different so generators aren't expended!
        self.candidate_spaces = list(map(deepcopy, self.candidate_spaces))

        # Preallocates internal data structures
        self.child_context_sets = [None] * self.arity
        for i in range(self.arity):
            self.child_context_sets[i] = set()

        super(CrossContextCandidateExtractorUDF, self).__init__(**kwargs)

    def apply(self, context_list, clear, split, window_size, thresholds, exclusive=False, **kwargs):
        # Generate TemporaryContexts that are children of the contexts using the candidate_space and filtered
        # by the Matcher

        """
        :param window_size: Number of adjacent contexts to extract candidates from at once
        :param thresholds: Maximum numbers of detections for each matcher
        """

        # if 'window_size' not in kwargs:
        #     window_size = 1
        # elif isinstance(kwargs['window_size'], int) and kwargs['window_size'] >= 1:
        #     window_size = kwargs['window_size']
        # else:
        #     raise ValueError('Window size must be an integer greater than 0.')
            
        # if 'thresholds' not in kwargs:
        #     thresholds = [5 for i in range(self.arity)]
        # elif isinstance(kwargs['thresholds'], list) and all(isinstance(i, int) and i >= 1 for i in kwargs['thresholds']):
        #     thresholds = kwargs['thresholds']
        # else:
        #     raise ValueError('Thresholds must be a list of integers greater than 0.')

        list_size = len(context_list)
        queue = deque([], window_size)  

        #Load queue
        for i in range(min(window_size, list_size)):
            context = context_list[i]
            temp_list = []
            for j in range(self.arity):
                temp_cand_space = deepcopy(self.candidate_spaces[j])
                temp_set = set()

                for count, tc in enumerate(self.matchers[j].apply(temp_cand_space.apply(context))):
                    if count >= thresholds[j]:
                        temp_set.clear()
                        break
                    else:      
                        temp_set.add(tc)
                           
                for tc in temp_set:
                    tc.load_id_or_insert(self.session)
                                           
                temp_list.append(temp_set)                                
            queue.append(temp_list)
                    
        #Iterate over contexts and generate candidates for each window configuration
        for context_index in range(list_size-1):
                                           
            for i in range(self.arity):
                self.child_context_sets[i].clear()
                                           
            for contex_set_list in queue:
                for i in range(self.arity):
                    self.child_context_sets[i].update(contex_set_list[i])
                                   
            lead_context_set_list = queue.popleft()
            
            #Generate potential candidate combinations
            potential_candidates = []  
            
            for i in range(self.arity):
                concatenated_set = self.child_context_sets[i]
                self.child_context_sets[i] = lead_context_set_list[i]
                potential_candidates.extend(product(*[enumerate(child_contexts) for child_contexts in self.child_context_sets]))  
                self.child_context_sets[i] = concatenated_set.difference(lead_context_set_list[i])

            # Generates and persists candidates
            extracted = set()
            candidate_args = {'split': split}
            for args in potential_candidates:

                if any([(arg.char_end - arg.char_start) <= 1 for (index,arg) in args]):
                    continue

                # TODO: Make this work for higher-order relations
                if self.arity == 2:

                    ai, a = args[0]
                    bi, b = args[1]

                    #Uncomment to only extract relations across window_size sentences
                    if exclusive and abs(a.sentence.position - b.sentence.position) != window_size - 1:
                        continue

                    # Check for self-joins, "nested" joins (joins from span to its subspan), and flipped duplicate
                    # "symmetric" relations. For symmetric relations, if mentions are of the same type, maintain
                    # their order in the sentence.
                    if not self.self_relations and a == b:
                        continue
                    elif not self.nested_relations and (a in b or b in a):
                        continue
                    elif not self.symmetric_relations and ((b, a) in extracted):
                    	continue

                    # Keep track of extracted
                    extracted.add((a,b))

                # Assemble candidate arguments
                for i, arg_name in enumerate(self.candidate_class.__argnames__):
                    candidate_args[arg_name + '_id'] = args[i][1].id

                # Checking for existence
                if not clear:
                    q = select([self.candidate_class.id])
                    for key, value in iteritems(candidate_args):
                        q = q.where(getattr(self.candidate_class, key) == value)
                    candidate_id = self.session.execute(q).first()
                    if candidate_id is not None:
                        continue

                # Add Candidate to session
                yield self.candidate_class(**candidate_args)
                                           
            #Load next item into the queue
            window_offset = context_index + window_size
            if window_offset < list_size:
                new_context = context_list[window_offset]
                for j in range(self.arity):
                    temp_cand_space = deepcopy(self.candidate_spaces[j])
                    lead_context_set_list[j].clear()
                    for count, tc in enumerate(self.matchers[j].apply(temp_cand_space.apply(new_context))):
                        if count >= thresholds[j]:
                            lead_context_set_list[j].clear()
                            break
                        else:
                            lead_context_set_list[j].add(tc)

                    for tc in lead_context_set_list[j]:
                        tc.load_id_or_insert(self.session)
                    
                queue.append(lead_context_set_list)


class CandidateSpace(object):
    """
    Defines the **space** of candidate objects
    Calling _apply(x)_ given an object _x_ returns a generator over candidates in _x_.
    """
    def __init__(self):
        pass

    def apply(self, x):
        raise NotImplementedError()


class Ngrams(CandidateSpace):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a Sentence _x_,
    indexing by **character offset**.
    """
    def __init__(self, n_max=5, split_tokens=('-', '/')):
        CandidateSpace.__init__(self)
        self.n_max     = n_max
        self.split_rgx = r'('+r'|'.join(split_tokens)+r')' if split_tokens and len(split_tokens) > 0 else None

    def apply(self, context):

        # These are the character offset--**relative to the sentence start**--for each _token_
        offsets = context.char_offsets

        # Loop over all n-grams in **reverse** order (to facilitate longest-match semantics)
        L    = len(offsets)
        seen = set()
        for l in range(1, self.n_max+1)[::-1]:
            for i in range(L-l+1):
                w     = context.words[i+l-1]
                start = offsets[i]
                end   = offsets[i+l-1] + len(w) - 1
                ts    = TemporarySpan(char_start=start, char_end=end, sentence=context)
                if ts not in seen:
                    seen.add(ts)
                    yield ts

                # Check for split
                # NOTE: For simplicity, we only split single tokens right now!
                if l == 1 and self.split_rgx is not None and end - start > 0:
                    m = re.search(self.split_rgx, context.text[start-offsets[0]:end-offsets[0]+1])
                    if m is not None and l < self.n_max + 1:
                        ts1 = TemporarySpan(char_start=start, char_end=start + m.start(1) - 1, sentence=context)
                        if ts1 not in seen:
                            seen.add(ts1)
                            yield ts
                        ts2 = TemporarySpan(char_start=start + m.end(1), char_end=end, sentence=context)
                        if ts2 not in seen:
                            seen.add(ts2)
                            yield ts2


class PretaggedCandidateExtractor(UDFRunner):
    """UDFRunner for PretaggedCandidateExtractorUDF"""
    def __init__(self, candidate_class, entity_types, self_relations=False,
     nested_relations=False, symmetric_relations=True, entity_sep='~@~'):
        super(PretaggedCandidateExtractor, self).__init__(
            PretaggedCandidateExtractorUDF, candidate_class=candidate_class,
            entity_types=entity_types, self_relations=self_relations,
            nested_relations=nested_relations, entity_sep=entity_sep,
            symmetric_relations=symmetric_relations,
        )

    def apply(self, xs, split=0, **kwargs):
        super(PretaggedCandidateExtractor, self).apply(xs, split=split, **kwargs)

    def clear(self, session, split, **kwargs):
        session.query(Candidate).filter(Candidate.split == split).delete()


class PretaggedCandidateExtractorUDF(UDF):
    """
    An extractor for Sentences with entities pre-tagged, and stored in the entity_types and entity_cids
    fields.
    """
    def __init__(self, candidate_class, entity_types, self_relations=False, nested_relations=False, symmetric_relations=False, entity_sep='~@~', **kwargs):
        self.candidate_class     = candidate_class
        self.entity_types        = entity_types
        self.arity               = len(entity_types)
        self.self_relations      = self_relations
        self.nested_relations    = nested_relations
        self.symmetric_relations = symmetric_relations
        self.entity_sep          = entity_sep

        super(PretaggedCandidateExtractorUDF, self).__init__(**kwargs)

    def apply(self, context, clear, split, check_for_existing=True, **kwargs):
        """Extract Candidates from a Context"""
        # For now, just handle Sentences
        if not isinstance(context, Sentence):
            raise NotImplementedError("PretaggedCandidateExtractor is currently only implemented for Sentence contexts.")

        # Do a first pass to collect all mentions by entity type / cid
        entity_idxs = dict((et, defaultdict(list)) for et in set(self.entity_types))
        L = len(context.words)
        for i in range(L):
            if context.entity_types[i] is not None:
                ets  = context.entity_types[i].split(self.entity_sep)
                cids = context.entity_cids[i].split(self.entity_sep)
                for et, cid in zip(ets, cids):
                    if et in entity_idxs:
                        entity_idxs[et][cid].append(i)

        # Form entity Spans
        entity_spans = defaultdict(list)
        entity_cids  = {}
        for et, cid_idxs in iteritems(entity_idxs):
            for cid, idxs in iteritems(entity_idxs[et]):
                while len(idxs) > 0:
                    i          = idxs.pop(0)
                    char_start = context.char_offsets[i]
                    char_end   = char_start + len(context.words[i]) - 1
                    while len(idxs) > 0 and idxs[0] == i + 1:
                        i        = idxs.pop(0)
                        char_end = context.char_offsets[i] + len(context.words[i]) - 1

                    # Insert / load temporary span, also store map to entity CID
                    tc = TemporarySpan(char_start=char_start, char_end=char_end, sentence=context)
                    tc.load_id_or_insert(self.session)
                    entity_cids[tc.id] = cid
                    entity_spans[et].append(tc)

        # Generates and persists candidates
        candidate_args = {'split' : split}
        for args in product(*[enumerate(entity_spans[et]) for et in self.entity_types]):

            # TODO: Make this work for higher-order relations
            if self.arity == 2:
                ai, a = args[0]
                bi, b = args[1]

                # Check for self-joins, "nested" joins (joins from span to its subspan), and flipped duplicate
                # "symmetric" relations
                if not self.self_relations and a == b:
                    continue
                elif not self.nested_relations and (a in b or b in a):
                    continue
                elif not self.symmetric_relations and ai > bi:
                    continue
            #Assemble candidate arguments
            for i, arg_name in enumerate(self.candidate_class.__argnames__):
                candidate_args[arg_name + '_id'] = args[i][1].id
                candidate_args[arg_name + '_cid'] = entity_cids[args[i][1].id]

            # Checking for existence
            if check_for_existing:
                q = select([self.candidate_class.id])
                for key, value in iteritems(candidate_args):
                    q = q.where(getattr(self.candidate_class, key) == value)
                candidate_id = self.session.execute(q).first()
                if candidate_id is not None:
                    continue

            # Add Candidate to session
            yield self.candidate_class(**candidate_args)


class CrossContextPretaggedCandidateExtractor(UDFRunner):
    """UDFRunner for CrossContextPretaggedCandidateExtractorUDF"""
    def __init__(self, candidate_class, entity_types, self_relations=False,
     nested_relations=False, symmetric_relations=True, entity_sep='~@~'):
        super(CrossContextPretaggedCandidateExtractor, self).__init__(
            CrossContextPretaggedCandidateExtractorUDF, candidate_class=candidate_class,
            entity_types=entity_types, self_relations=self_relations,
            nested_relations=nested_relations, entity_sep=entity_sep,
            symmetric_relations=symmetric_relations,
        )

    def apply(self, xs, split=0, **kwargs):
        super(CrossContextPretaggedCandidateExtractor, self).apply(xs, split=split, **kwargs)

    def clear(self, session, split, **kwargs):
        session.query(Candidate).filter(Candidate.split == split).delete()


class CrossContextPretaggedCandidateExtractorUDF(UDF):
    """
    An extractor for Sentences with entities pre-tagged, and stored in the entity_types and entity_cids
    fields.
    """
    def __init__(self, candidate_class, entity_types, self_relations=False, nested_relations=False, symmetric_relations=False, entity_sep='~@~', **kwargs):
        self.candidate_class     = candidate_class
        self.entity_types        = entity_types
        self.arity               = len(entity_types)
        self.self_relations      = self_relations
        self.nested_relations    = nested_relations
        self.symmetric_relations = symmetric_relations
        self.entity_sep          = entity_sep

        # Preallocates internal data structures
        self.cand_sets = [None] * self.arity
        for i in range(self.arity):
            self.cand_sets[i] = set()

        super(CrossContextPretaggedCandidateExtractorUDF, self).__init__(**kwargs)

    def apply(self, context_list, clear, split, window_size, exclusive=False, check_for_existing=True, **kwargs):
        """Extract Candidates from a list of contexts"""
        # For now, just handle Sentences
        if not isinstance(context_list[0], Sentence):
            raise NotImplementedError("CrossContextPretaggedCandidateExtractor is currently only implemented for Sentence contexts.")

        # if 'window_size' not in kwargs:
        #     window_size = 1
        # elif isinstance(kwargs['window_size'], int) and kwargs['window_size'] >= 1:
        #     window_size = kwargs['window_size']
        # else:
        #     raise ValueError('Window size must be an integer greater than 0.')
            
        # if 'thresholds' not in kwargs:
        #     thresholds = [5 for i in range(self.arity)]
        # elif isinstance(kwargs['thresholds'], list) and all( isinstance(i, int) and i >= 1 for i in kwargs['thresholds']):
        #     thresholds = kwargs['thresholds']
        # else:
        #     raise ValueError('Thresholds must be a list of integers greater than 0.')

        list_size = len(context_list)
        queue = deque([], window_size)
        entity_cids  = {}  

        #Load queue
        for i in range(min(window_size, list_size)):

            context = context_list[i]
            # Do a first pass to collect all mentions by entity type / cid
            entity_idxs = dict((et, defaultdict(list)) for et in set(self.entity_types))
            L = len(context.words)
            for i in range(L):
	            if context.entity_types[i] is not None:
	                ets  = context.entity_types[i].split(self.entity_sep)
	                cids = context.entity_cids[i].split(self.entity_sep)
	                for et, cid in zip(ets, cids):
	                    if et in entity_idxs:
	                        entity_idxs[et][cid].append(i)

	        # Form entity Spans
            entity_spans = defaultdict(list)
            for et, cid_idxs in iteritems(entity_idxs):
	            for cid, idxs in iteritems(entity_idxs[et]):
	                while len(idxs) > 0:
	                    i          = idxs.pop(0)
	                    char_start = context.char_offsets[i]
	                    char_end   = char_start + len(context.words[i]) - 1
	                    while len(idxs) > 0 and idxs[0] == i + 1:
	                        i        = idxs.pop(0)
	                        char_end = context.char_offsets[i] + len(context.words[i]) - 1

	                    # Insert / load temporary span, also store map to entity CID
	                    tc = TemporarySpan(char_start=char_start, char_end=char_end, sentence=context)
	                    tc.load_id_or_insert(self.session)
	                    entity_cids[tc.id] = cid
	                    entity_spans[et].append(tc)
            queue.append(entity_spans)


	    #queue of dictonaries 

        for context_index in range(list_size):
            for i in range(self.arity):
                self.cand_sets[i].clear()


            #May have to concatenate/merge entity_cids to one dic
            for entity_spans in queue:
                for i, et in enumerate(self.entity_types):
                    self.cand_sets[i].update(entity_spans[et])

            lead_entity_span = queue.popleft()
            potential_candidates = []

            for i, et in enumerate(self.entity_types):
                concatenated_set = self.cand_sets[i]
                self.cand_sets[i] = lead_entity_span[et]
                potential_candidates.extend(product(*[enumerate(cand_set) for cand_set in self.cand_sets]))
                self.cand_sets[i] = concatenated_set.difference(lead_entity_span[et])


	        # Generates and persists candidates
            candidate_args = {'split' : split}

            for args in potential_candidates:

	            # TODO: Make this work for higher-order relations
                if self.arity == 2:
                    ai, a = args[0]
                    bi, b = args[1]

                    #Uncomment to only extract relations across window_size sentences
                    if exclusive and abs(a.sentence.position - b.sentence.position) != window_size - 1:
                        continue

	                # Check for self-joins, "nested" joins (joins from span to its subspan), and flipped duplicate
	                # "symmetric" relations
                    if not self.self_relations and a == b:
                        continue
                    elif not self.nested_relations and (a in b or b in a):
                        continue
                    elif not self.symmetric_relations and ai > bi:
                        continue
                    
                for i, arg_name in enumerate(self.candidate_class.__argnames__):
                    candidate_args[arg_name + '_id'] = args[i][1].id
                    candidate_args[arg_name + '_cid'] = entity_cids[args[i][1].id]

	            # Checking for existence
                if check_for_existing:
                    q = select([self.candidate_class.id])
                    for key, value in iteritems(candidate_args):
                        q = q.where(getattr(self.candidate_class, key) == value)
                        candidate_id = self.session.execute(q).first()
                        if candidate_id is not None:
                            continue

	            # Add Candidate to session
                yield self.candidate_class(**candidate_args)

	        #Load next item into the queue
            window_offset = context_index + window_size
            if window_offset < list_size:
                context = context_list[window_offset]

		        # Do a first pass to collect all mentions by entity type / cid
                entity_idxs = dict((et, defaultdict(list)) for et in set(self.entity_types))
                L = len(context.words)
                for i in range(L):
                    if context.entity_types[i] is not None:
                        ets  = context.entity_types[i].split(self.entity_sep)
                        cids = context.entity_cids[i].split(self.entity_sep)
                        for et, cid in zip(ets, cids):
                            if et in entity_idxs:
                                entity_idxs[et][cid].append(i)

		        # Form entity Spans
                entity_spans = defaultdict(list)
                for et, cid_idxs in iteritems(entity_idxs):
                    for cid, idxs in iteritems(entity_idxs[et]):
                        while len(idxs) > 0:
                            i          = idxs.pop(0)
                            char_start = context.char_offsets[i]
                            char_end   = char_start + len(context.words[i]) - 1
                            while len(idxs) > 0 and idxs[0] == i + 1:
                                i        = idxs.pop(0)
                                char_end = context.char_offsets[i] + len(context.words[i]) - 1

		                    # Insert / load temporary span, also store map to entity CID
                            tc = TemporarySpan(char_start=char_start, char_end=char_end, sentence=context)
                            tc.load_id_or_insert(self.session)
                            entity_cids[tc.id] = cid
                            entity_spans[et].append(tc)

                queue.append(entity_spans)





