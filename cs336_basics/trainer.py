import os
import time
from turtle import st
from numpy.random import default_rng
import regex as re
import numpy as np
import multiprocessing as mp

from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, BinaryIO


# GPT-2 pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETrainer:
    def __init__(self) -> None:
        self.vocab = {}
        self.merges = []


    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> List[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        
        mini_chunk_size = 4096 # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                
                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                
                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
        
        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _get_word_freqs_chunk(
        self,
        chunk: str,
        special_tokens: List[str],
    ) -> Dict[Tuple[bytes, ...], int]:
        """
        Process a chunk of text and return the word frequencies
        """
        word_freqs = defaultdict(int)

        # Split on special tokens to avoid merging across boundaries
        pattern = r"|".join(re.escape(token) for token in special_tokens)
        chunk = re.split(f"({pattern})", chunk)
        chunk = [part for part in chunk if part and part not in special_tokens]

        for text in chunk:
            if not text.strip():
                continue

            # Pre-tokenize using regex pattern
            for match in re.finditer(PAT, text):
                word = match.group()
                # Convert to bytes and then to tuple of individual bytes
                word_freqs[tuple(word.encode('utf-8'))] += 1

        return dict(word_freqs)

    def _process_chunk_parallel(self, args):
        """
        Process a single chunk for multiprocessing
        """
        start, end, input_path, special_tokens = args
        # Read chunk from file
        with open(input_path, 'rb') as f:
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk = chunk_bytes.decode('utf-8', errors='ignore')
        
        return self._get_word_freqs_chunk(chunk, special_tokens)

    def _merge_word_freqs(
        self,
        freq_dicts: List[Dict],
    ) -> Dict[Tuple[bytes, ...], int]:
        """
        Merge frequency dictionaties from multiple processes
        """
        combined_freqs = defaultdict(int)
        for freq_dict in freq_dicts:
            for word, freq in freq_dict.items():
                combined_freqs[word] += freq
        return dict(combined_freqs)

    def _get_pairs(
        self,
        word_freqs: Dict[Tuple[bytes, ...], int],
    ) -> Dict[Tuple[bytes, bytes], int]:
        """
        Get all pairs and their frequencies
        """
        pairs = defaultdict(int)

        for word, freq in word_freqs.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        
        return dict(pairs)

    def _merge_vocab(
        self,
        pair: Tuple[bytes, bytes],
        word_freqs: Dict[Tuple[bytes, ...], int]
    ) -> Dict[Tuple[bytes, ...], int]:
        """
        Merge the most frequent pair into the vocabulary
        """
        new_word_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                # Check if the current pair matches the pair to merge
                if i < len(word) - 1 and word[i:i+2] == pair:
                    new_word.append(pair)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] += freq
        
        return dict(new_word_freqs)
    
    
    def train_bpe(
        self,
        input_path: str, 
        vocab_size: int, 
        special_tokens: List[str]
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Train a BPE tokenizer.
        """
        if vocab_size < 256 + len(special_tokens):
            raise ValueError("Vocab size is too small")

        # initialize vocabulary    
        vocab = {i: bytes([i]) for i in range(256)}

        # add special tokens to vocabulary
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            vocab[len(vocab)] = token_bytes
        
        # Pre-tokenize and frequency statistics
        # Open input file in binary mode 
        num_process = mp.cpu_count()
        with open(input_path, 'rb', encoding='utf-8') as f:
            desired_num_chunks = num_process * 10 # each process will handle 10 chunks
            split_special_token = special_tokens[0].encode('utf-8') if special_tokens else b"<|endoftext|>"
            # Get chunk boundaries
            boundaries = self._find_chunk_boundaries(f, desired_num_chunks, split_special_token)

            # Prepare arguments for parallel processing
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append((start, end, input_path, special_tokens))
        
        # Process chunks in parallel
        with mp.Pool(num_process) as pool:
            freq_dicts = pool.map(self._process_chunk_parallel, chunk_args)
        
        # Merge frequency dictionaries
        word_freqs = self._merge_word_freqs(freq_dicts)
        print(f"Pre-tokenization complete. {len(word_freqs)} unique words found.")

        # BPE training
        merges = []
        while len(vocab) < vocab_size:
            pairs = self._get_pairs(word_freqs)

            if not pairs:
                break

            # Find the most frequent pair (break ties lexicographically)
            most_frequent_pair = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]

            # Merge the vocabulary
            word_freqs = self._merge_vocab(most_frequent_pair, word_freqs)

            # Add merge to list and new token to vocabulary
            merges.append(most_frequent_pair)
            vocab[len(vocab)] = most_frequent_pair

        print(f"BPE training complete. {len(vocab)} tokens in vocabulary.")

        self.vocab = vocab
        self.merges = merges

        return vocab, merges

def run_train_bpe(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Adapter function for the BPE training
    """
    trainer = BPETrainer()
    return trainer.train_bpe(input_path, vocab_size, special_tokens)