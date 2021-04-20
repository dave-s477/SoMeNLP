first_char_lower = lambda s: s[:1].lower() + s[1:] if s else ''      

def pan_top_1(x):
    '''use <> software'''
    left_context = ['use']
    right_context = ['software']    
    left_lemmas = x.sentence.get_left_tokens(x.start_idx, size=1, style='lemma')
    right_lemmas = x.sentence.get_right_tokens(x.end_idx, size=1, style='lemma')
    if not left_lemmas or not right_lemmas or len(left_context) != len(left_lemmas) or len(right_context) != len(right_lemmas):
        return False
    for cont, feat in zip(left_context, left_lemmas):
        if cont != feat:
            return False
    for cont,feat in zip(right_context, right_lemmas):
        if cont != feat:
            return False
    return True

def pan_top_2(x):
    '''perform use <>'''
    left_context = ['perform', 'use']
    left_lemmas = x.sentence.get_left_tokens(x.start_idx, size=2, style='lemma')
    if len(left_context) != len(left_lemmas):
        return False
    for c,l in zip(left_context, left_lemmas):
        if c != l:
            return False
    return True

def pan_top_3(x):
    '''be perform use <>'''
    left_context = ['be', 'perform', 'use']
    left_lemmas = x.sentence.get_left_tokens(x.start_idx, size=3, style='lemma')
    if len(left_context) != len(left_lemmas):
        return False
    for c,l in zip(left_context, left_lemmas):
        if c != l:
            return False
    return True

def pan_top_4(x):
    '''analysis be perform use <>'''
    left_context = ['analysis', 'be', 'perform', 'use']
    left_lemmas = x.sentence.get_left_tokens(x.start_idx, size=4, style='lemma')
    if len(left_context) != len(left_lemmas):
        return False
    for c,l in zip(left_context, left_lemmas):
        if c != l:
            return False
    return True

def pan_top_5(x):
    '''analyze use <>'''
    left_context_ae = ['analyze', 'use']
    left_context_be = ['analyse', 'use']
    left_lemmas = x.sentence.get_left_tokens(x.start_idx, size=2, style='lemma')
    if len(left_context_ae) != len(left_lemmas):
        return False
    for c_ae, c_be, l in zip(left_context_ae, left_context_be, left_lemmas):
        if c_ae != l and c_be != l:
            return False
    return True

def pan_top_6(x):
    '''analysis be perform with <>'''
    left_context = ['analysis', 'be', 'perform', 'with']
    left_lemmas = x.sentence.get_left_tokens(x.start_idx, size=4, style='lemma')
    if len(left_context) != len(left_lemmas):
        return False
    for c,l in zip(left_context, left_lemmas):
        if c != l:
            return False
    return True

def pan_top_7(x):
    '''<> statistical software'''
    right_context = ['statistical', 'software']
    right_lemmas = x.sentence.get_right_tokens(x.end_idx, size=2, style='lemma')
    if not right_lemmas or len(right_context) != len(right_lemmas):
        return False
    for c,l in zip(right_context, right_lemmas):
        if c != l:
            return False
    return True

def pan_top_8(x):
    '''<> software be use'''
    right_context = ['software', 'be', 'use']
    right_lemmas = x.sentence.get_right_tokens(x.end_idx, size=3, style='lemma')
    if not right_lemmas or len(right_context) != len(right_lemmas):
        return False
    for c,l in zip(right_context, right_lemmas):
        if c != l:
            return False
    return True

def text_is_in_brackets(x):
    left_context = x.sentence.get_left_tokens(x.start_idx, 1, style='plain')
    right_context = x.sentence.get_right_tokens(x.end_idx, 1, style='plain')
    if len(x.base_span) != 1 and left_context and right_context and left_context[0] in ['(', '[', '{'] and right_context[0] in [')', ']', '}']:
        return True
    else:
        return False

def developer(x):
    '''software developer match'''
    left_context = x.sentence.get_left_tokens(x.start_idx, 1, style='plain')
    right_context = x.sentence.get_right_tokens(x.end_idx, 1, style='plain')
    if len(x.base_span) != 1 and left_context and right_context and left_context[0] == '(' and right_context[0] == ')':
        for tok in x.tokens:
            if tok.lower() in ['inc.', 'ltd.', 'corp.', 'apply', 'inc', 'ltd', 'corp']:
                return True
    return False
