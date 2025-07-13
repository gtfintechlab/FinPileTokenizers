import numpy as np
from transformers import AutoTokenizer
import argparse
import glob
import os
from fsiltok.jsonl import JSONLFileReader
from fsiltok.builder import DatasetBuilder
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

tokenizer_ : AutoTokenizer = None

def init_tokenizer(tokenizer_name: str, use_fast: bool = True):
    global tokenizer_
    tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast)
    print(f"Tokenizer initialized: {tokenizer_.name_or_path}, EOS token: {tokenizer_.eos_token}")

def tokenize(document: str, append_eos: bool, dtype: np.dtype) -> np.ndarray:
    global tokenizer_

    tokens = tokenizer_.encode(document)
    if append_eos:
        eos_token_id = tokenizer_.eos_token_id
        if eos_token_id is not None and tokens[-1] != eos_token_id:
            tokens.append(eos_token_id)

    return np.array(tokens, dtype=dtype), len(tokens)

def main():
    parser = argparse.ArgumentParser(description="Run the fsiltok tokenizer.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for the output files.")
    parser.add_argument("--append-eos", action="store_true", help="Append EOS token to the output.")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads to use.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer to use.")
    parser.add_argument("--dtype", type=str, default="uint16", help="Data type for the dataset builder.")
    parser.add_argument("--text-field", type=str, default="text", help="Field in the JSONL file to read text from.")
    parser.add_argument("--use-slow", action="store_true", help="Use slow tokenizer if available.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist.")
    if os.path.isdir(args.input):
        paths = (
            glob.glob(os.path.join(args.input, "*.jsonl.gz"))
            + glob.glob(os.path.join(args.input, "*.json.gz"))
        )
    else:
        paths = [args.input]

    if not paths:
        raise ValueError(f"No valid input files found in {args.input}.")
    
    assert args.tokenizer is not None, "Tokenizer must be specified."

    dtype : np.dtype = np.dtype(args.dtype)
    assert dtype in [np.uint16, np.uint32, np.uint64], "Unsupported data type. Use uint16, uint32, or uint64."


    global tokenizer_

    stream = iter(JSONLFileReader(args.input, textField=args.text_field, threads=0))
    builder = DatasetBuilder(args.prefix, dtype=dtype)

    tokenize_partial = partial(tokenize, append_eos=args.append_eos, dtype=dtype)

    stream = tqdm(stream, desc="Read documents", unit=" documents", position=0, leave=False)

    if args.threads > 1:
        pool = Pool(args.threads, initializer=init_tokenizer, initargs=(args.tokenizer, args.use_fast))
        stream = pool.imap(tokenize_partial, stream, chunksize=1)
    else:
        init_tokenizer(args.tokenizer, use_fast=args.use_fast)
        stream = map(tokenize_partial, stream)

    stream = tqdm(stream, desc="Tokenized documents", unit=" documents", position=1, leave=False)
    for tokens, size in stream:
        if size > 0:
            builder.extend(tokens)

    builder.finalize()
    print(f"Dataset built successfully at {args.prefix}.bin with index {args.prefix}.idx")
    print(f"Total size: {builder._size} tokens.")

# Probably not needed - should be run with the pyproject entrypoint 'fsiltok [args]'
if __name__ == "__main__":
    main()