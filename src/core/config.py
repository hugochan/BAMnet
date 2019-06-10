
# Vocabulary
RESERVED_TOKENS = {'PAD': 0, 'UNK': 1}
RESERVED_ENTS = {'PAD': 0, 'UNK': 1}
RESERVED_ENT_TYPES = {'PAD': 0, 'UNK': 1}
RESERVED_RELS = {'PAD': 0, 'UNK': 1}

extra_vocab_tokens = ['alias', 'true', 'false', 'num', 'bool'] + \
    ['np', 'organization', 'date', 'number', 'misc', 'ordinal', 'duration', 'person', 'time', 'location'] + \
    ['__np__', '__organization__', '__date__', '__number__', '__misc__', '__ordinal__', '__duration__', '__person__', '__time__', '__location__']

extra_rels = ['alias']
extra_ent_types = ['num', 'bool']


# BAMnet entity mention types
topic_mention_types = {'person', 'organization', 'location', 'misc'}
# delex_mention_types = {'date', 'time', 'ordinal', 'number'}
delex_mention_types = {'date', 'ordinal', 'number'}
constraint_mention_types = delex_mention_types
