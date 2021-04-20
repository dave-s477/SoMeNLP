def upper_cased(x):
    if x.isalpha() and x.isupper():
        return True
    else:
        return False

def first_char_upper(x):
    if x.isalpha() and x[0].isupper():
        if all(c.islower() for c in x[1:]):
            return True
    return False

def mixed_case(x):
    if x.isalpha():
        if any([c.isupper() for c in x[1:]]) and any([c.islower() for c in x]):
            return True
    return False

def lower_case(x):
    if x.isalpha() and x.islower():
        return True
    else:
        return False