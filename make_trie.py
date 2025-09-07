import json
import os
import gzip

WORD_MARKER = 26
RETURN_MARKER = 27

def to_byte(char: str) -> int:
    return ord(char) - ord('a')

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

    def to_dict(self):
        out_dict = {c: child.to_dict() for c, child in self.children.items()}
        if self.is_end_of_word:
            out_dict['$'] = self.is_end_of_word
        return out_dict

    def add_word(self, word: str):
        if not word:
            self.is_end_of_word = True
            return
        char, rest = word[0], word[1:]
        if char not in self.children:
            self.children[char] = TrieNode()
        self.children[char].add_word(rest)

    def to_compact_repr(self):
        result = bytearray()
        if not self.children and not self.is_end_of_word:
            raise ValueError("Empty node cannot be serialized")
        if self.is_end_of_word:
            result.append(WORD_MARKER)
        for char, child in sorted(self.children.items()):
            result.append(to_byte(char))
            result.extend(child.to_compact_repr())
            result.append(RETURN_MARKER)
        return result

def bytes_to_debug_string(byte_array: bytearray) -> str:
    result = []
    for byte in byte_array:
        if byte == WORD_MARKER:
            result.append('$')
        elif byte == RETURN_MARKER:
            result.append('^')
        else:
            result.append(chr(byte + ord('a')))
    return ''.join(result) + '^'

def load_trie_from_compact_repr(compact_repr: bytearray) -> TrieNode:
    root = TrieNode()
    stack = [root]
    i = 0
    for byte in compact_repr:
        current_node = stack[-1]
        if byte == WORD_MARKER:
            current_node.is_end_of_word = True
            continue
        elif byte == RETURN_MARKER:
            stack.pop()
            continue
        else:
            char = chr(byte + ord('a'))
            new_node = TrieNode()
            current_node.children[char] = new_node
            stack.append(new_node)
            current_node = new_node
            i += 1
    return root

def load_words(filepath: str, limit=None):
    count =0
    with open(filepath, 'r') as file:
        for line in file:
            word = line.strip()
            if len(word) < 3 or len(word) > 25:
                continue
            yield word.lower()
            count += 1
            if limit is not None and count >= limit:
                break

def write_to_gzipped_file(data: bytearray, filepath: str):
    with gzip.open(filepath, 'wb') as f:
        f.write(data)
    # print file size
    print(f"Wrote {os.path.getsize(filepath)} bytes to {filepath}")

def main():
    words = list(load_words("words_alpha.txt"))
    trie = TrieNode()
    for word in words:
        trie.add_word(word)
    compact_repr = trie.to_compact_repr()
    print(f"Total words: {len(words)}")
    print(f"Trie size (bytes): {len(compact_repr)}")
    print(f"Preview: {bytes_to_debug_string(compact_repr[:100])}")

    write_to_gzipped_file(compact_repr, "trie_data.gz")

    json_repr = json.dumps(trie.to_dict())
    write_to_gzipped_file(bytearray(json_repr, 'utf-8'), "trie_data.json.gz")

if __name__ == "__main__":
    main()