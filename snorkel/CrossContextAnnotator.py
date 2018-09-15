class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

from IPython.display import clear_output


class CrossContextAnnotator(UDFRunner):
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
        super(CrossContextAnnotator, self).__init__(CrossContextAnnotatorUDF,
                                                 candidate_class=candidate_class,
                                                 cspaces=cspaces,
                                                 matchers=matchers,
                                                 self_relations=self_relations,
                                                 nested_relations=nested_relations,
                                                 symmetric_relations=symmetric_relations)

    def apply(self, xs, split=0, **kwargs):
        super(CrossContextAnnotator, self).apply(xs, split=split, **kwargs)

    def clear(self, session, split, **kwargs):
        session.query(Candidate).filter(Candidate.split == split).delete()


class CrossContextAnnotatorUDF(UDF):
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

        super(CrossContextAnnotatorUDF, self).__init__(**kwargs)

    def apply(self, doc, clear, split, **kwargs):
        # Generate TemporaryContexts that are children of the contexts using the candidate_space and filtered
        # by the Matcher

        """
        :param window_size: Number of adjacent contexts to extract candidates from at once
        :param thresholds: Maximum numbers of detections for each matcher
        """

        context_name, context_list = doc

        if 'window_size' not in kwargs:
            window_size = 1
        elif isinstance(kwargs['window_size'], int) and kwargs['window_size'] >= 1:
            window_size = kwargs['window_size']
        else:
            raise ValueError('Window size must be an integer greater than 0.')
            
        if 'thresholds' not in kwargs:
            thresholds = [5 for i in range(self.arity)]
        elif isinstance(kwargs['thresholds'], list) and all( i >= 1 and isinstance(i, int) for i in kwargs['thresholds']):
            thresholds = kwargs['thresholds']
        else:
            raise ValueError('Thresholds must be a list of integers greater than 0.')

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


        with open("data/two_sentence_gold_labels.tsv", "a") as label_file:

                    
            #Iterate over contexts and generate candidates for each window configuration
            for context_index in range(list_size):
                                               
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


                        if abs(a.sentence.position - b.sentence.position) != window_size - 1:
                            continue

                        # Check for self-joins, "nested" joins (joins from span to its subspan), and flipped duplicate
                        # "symmetric" relations. For symmetric relations, if mentions are of the same type, maintain
                        # their order in the sentence.

                        word_a = a.sentence.text[a.char_start:a.char_end+1]
                        word_b = b.sentence.text[b.char_start:b.char_end+1]

                        if not self.self_relations and (a == b or word_a in word_b or word_b in word_a):
                            continue
                        elif not self.nested_relations and (a in b or b in a):
                            continue
                        elif not self.symmetric_relations and ((b, a) in extracted):
                            continue

                        # Keep track of extracted
                        extracted.add((a,b))

                        if a.sentence.position <= b.sentence.position:
                            first_mention = a
                            second_mention = b
                        else:
                            first_mention = b
                            second_mention = a


                        first_sent = first_mention.sentence.text[:first_mention.char_start] + color.BOLD + first_mention.sentence.text[first_mention.char_start:first_mention.char_end+1].upper() + color.END + first_mention.sentence.text[first_mention.char_end+1:] 
                        second_sent = second_mention.sentence.text[:second_mention.char_start] + color.BOLD + second_mention.sentence.text[second_mention.char_start:second_mention.char_end+1].upper() + color.END + second_mention.sentence.text[second_mention.char_end+1:]

                        potential_string = first_sent + color.BOLD + '  |||  ' + color.END + second_sent

                        #TODO: Implement O(log(n)) binary search
                        first_mention_idx = min(range(len(first_mention.sentence.char_offsets)), key=lambda i: abs(first_mention.sentence.char_offsets[i]-first_mention.char_start))
                        first_mention_length = first_mention.char_end - first_mention.char_start
                        first_mention_abs_start = first_mention.sentence.abs_char_offsets[first_mention_idx]
                        first_mention_abs_end = first_mention_abs_start + first_mention_length

                        second_mention_idx = min(range(len(second_mention.sentence.char_offsets)), key=lambda i: abs(second_mention.sentence.char_offsets[i]-second_mention.char_start))
                        second_mention_length = second_mention.char_end - second_mention.char_start
                        second_mention_abs_start = second_mention.sentence.abs_char_offsets[second_mention_idx]
                        second_mention_abs_end = second_mention_abs_start + second_mention_length

                        clear_output()
                        label = input(potential_string + '\n' + "Enter label (1/-1), quit to exit, or any key to skip candidate:  ")

                        if label == '1' or label == '-1':
                            first_mention_tag = context_name+"::span:"+str(first_mention_abs_start)+':'+str(first_mention_abs_end)
                            second_mention_tag = context_name+"::span:"+str(second_mention_abs_start)+':'+str(second_mention_abs_end)
                            label_file.write(first_mention_tag+'\t'+second_mention_tag+'\t'+label+'\n')
                        elif label == 'quit':
                            label_file.close()
                            break
                        else:
                            continue

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
            label_file.close()

            yield self.candidate_class(**candidate_args)
