SCOWL_PATH = '~/Documents/wordlist/scowl/final'

import json
import os

def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def make_scowl_word_list(output_dir, size, spelling, min_length=3, max_length=25, include_variants=True):
    """Generate a word list from SCOWL files.

    Args:
        size: int, the size of the SCOWL files to use (e.g. 35, 40, 50, 60, 70, 80, 90, 95, 98)
        spelling: str, the type of spelling to use (e.g. 'american', 'british', 'canadian', 'australian')
        min_length: int, minimum length of words to include
        max_length: int, maximum length of words to include
    Returns:
        A set of words.
    """
    word_set = set()
    scowl_dir = os.path.expanduser(SCOWL_PATH)
    if not os.path.isdir(scowl_dir):
        raise ValueError(f"SCOWL path {scowl_dir} does not exist or is not a directory.")
    
    """
    Include: 
     - base word list (e.g. english-words.{val}) - where val <= size
     - spelling variant (e.g. english-{spelling}.{val}) - where val <= size
    - if include_variants is True, include variant files (e.g. variant_[0-9]-words.{val}, {spelling}-variant_[0-9]-words.{val})
    """

    def add_words_from_file(filepath):
        if not os.path.isfile(filepath):
            return
        print(f"Reading words from {filepath}...")
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                word = line.strip().lower()
                if min_length <= len(word) <= max_length and word.isalpha() and is_ascii(word):
                    word_set.add(word)



    for val in range(10, size + 1, 5):
        base_file = os.path.join(scowl_dir, f'english-words.{val}')
        add_words_from_file(base_file)

        spelling_file = os.path.join(scowl_dir, f'english-{spelling}.{val}')
        add_words_from_file(spelling_file)

        if include_variants:
            for i in range(1, 6):
                variant_file = os.path.join(scowl_dir, f'variant_{i}-words.{val}')
                add_words_from_file(variant_file)

                spelling_variant_file = os.path.join(scowl_dir, f'{spelling}_variant_{i}-words.{val}')
                add_words_from_file(spelling_variant_file)

    # Write to output file
    output_filename = f'{spelling}_{size}.txt'
    output_filename = os.path.join(output_dir, output_filename)
    with open(output_filename, 'w') as out_file:
        for word in sorted(word_set):
            out_file.write(word + '\n')

SIZES = [35, 40, 50, 55, 60]
SPELLINGS = ['american', 'british', 'canadian', 'australian']
def main():
    os.makedirs('wordlists', exist_ok=True)
    for size in SIZES:
        for spelling in SPELLINGS:
            print(f"Generating word list for size {size} and spelling {spelling}...")
            make_scowl_word_list('wordlists', size, spelling, min_length=3, max_length=25, include_variants=True)
            print(f"Word list for size {size} and spelling {spelling} generated.")
    metadata = {
        'sizes': SIZES,
        'spellings': SPELLINGS,
        'min_length': 3,
        'max_length': 25,
        'include_variants': True
    }
    with open('wordlists/metadata.json', 'w') as meta_file:
        json.dump(metadata, meta_file, indent=4)

if __name__ == "__main__":
    main()