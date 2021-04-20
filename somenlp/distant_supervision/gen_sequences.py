from itertools import product

GEN_SYMBOLS = ['T', 'C', 'G', 'A']

def generate_triplets():
    """Generate all possible triples

    Returns:
        dictionary: dictionary with list of possible GEN triplets
    """
    gen_triplets = [''.join(seq) for seq in product(GEN_SYMBOLS, repeat = 3)]
    return {'gen_triplets': gen_triplets}