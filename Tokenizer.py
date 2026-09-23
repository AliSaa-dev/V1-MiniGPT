import json
from functools import lru_cache

import regex as re

class Tokenizer:
    def __init__(self, encoder_path, vocabulary_BPE_path):

        # Open the encoder file
        with open(encoder_path, "r", encoding="utf-8") as f:

            #Load the encoder as json
            self.encoder = json.load(f)

        # Load the decoder as reverse of encoder
        self.decoder = {
            value: key
            for key, value in self.encoder.items()
        }

        # Open BPE file
        with open(vocabulary_BPE_path, "r", encoding="utf-8") as f:

            # Get all of lines in BPE
            BPE = f.read()

        # Get all of combinations
        combinations = [
            tuple(line.split())
            for line in BPE.strip().split("\n")[1:-1]
        ]

        # Get each combination with its priority
        self.BPE_priorities = {
            combination: priority
            for priority, combination in enumerate(combinations)
        }

        # Initiate the byte encoder.
        self.byte_encoder = self.byte_to_unicode()

        # Create the byte decoder as reverse of byte encoder
        self.byte_decoder = {
            value: key
            for key, value in self.byte_encoder.items()
        }

        # Create split patterns
        self.patterns = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|
             \p{L}+|
             \p{N}+|
             [^\s\p{L}\p{N}]+|
             \s+(?!\S)|
             \s+
            """,
            re.VERBOSE
        )

        self.cache = {}
        self.vocab_size = len(self.encoder)

    # Creating the original reversible byte-level mapper used by OpenAI in GPT models,
    # Every byte from 0 to 255 gets unique character.
    # Creating the function as static and using LRU_CACHE
    @staticmethod
    @lru_cache()
    def byte_to_unicode():

        # Start with byte values that already have convenient Unicode characters
        bs = (
                list(range(ord("!"), ord("~") + 1))
                + list(range(ord("¡"), ord("¬") + 1))
                + list(range(ord("®"), ord("ÿ") + 1))
        )

        cs = bs[:]
        n = 0

        # Add the remaining byte values and assign them unused Unicode code points
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)

                # Turn the rest to character unicodes
                cs.append(2 ** 8 + b)

                n += 1


        # Encode convert the unicodes to characters
        cs  =[chr(n) for n in cs]

        return dict(zip(bs, cs))

    # Create a static function for getting token pairs in each word
    @staticmethod
    def get_pairs(word):
        pairs = set()

        if len(word) < 2:
            return pairs

        previous = word[0]

        # Getting every possible pairs in the word
        for character in word[1:]:
            pairs.add(
                (previous, character)
            )
            previous = character

        return pairs

    def BPE(self, token):

        # Get the token from the cache
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self.get_pairs(word)

        # Return unchanged if the tokens are invalid
        if not pairs:
            return token

        while True:

            # Get the best priority possible as bigram
            bigram = min(
                pairs,
                key=lambda pair: self.BPE_priorities.get(
                    pair, float("inf")
                )
            )

            if bigram not in self.BPE_priorities:
                break

            first, second = bigram
            combined = []
            i = 0

            while i < len(word):
                try:

                    # Add the remaining byte values and assign them unused Unicode code points
                    j = word.index(first, i)

                    # Copy everything before the matching pair
                    combined.extend(word[i:j])

                    i = j

                except ValueError:
                    combined.extend(word[i:])
                    break

                # Combine if the current two elements match the selected pair
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    combined.append(first + second)
                    i += 2

                # Otherwise, copy the current element unchanged
                else:
                    combined.append(word[i])
                    i += 1

            word = tuple(combined)

            if len(word) == 1:
                break

            pairs = self.get_pairs(word)

        # Join the final BPE symbols into a space-separated token string
        result = " ".join(pairs)

        # Cache all the words as needed
        self.cache[token] = result

        return result

    def encode(self, text):

        token_ids = []

        # Split the text over the initiated patterns
        for token in self.patterns.findall(text):

            # Reverse the token with BPE as UTF-8
            token = "".join(
                self.byte_encoder[b]
                for b in token.encode("utf-8")
            )

            # Encode token with BPE
            BPE = self.BPE(token).split(" ")

            # Convert the token to UTF-8 bytes, then map each byte to a Unicode character
            for BPE_token in BPE:
                token_ids.append(self.encoder[BPE_token])

        return token_ids

    def decode(self, token_ids):

        # Decode the text and add the characters together
        text = "".join(
            self.decoder[token_id]
            for token_id in token_ids
        )

        # Convert the Unicode characters back into their original byte values
        byte_data = bytearray(
            self.byte_decoder[c]
            for c in text
        )

        # Decode the text as byte array to UTF-8
        return byte_data.decode(
            "utf-8",
            errors="replace"
        )