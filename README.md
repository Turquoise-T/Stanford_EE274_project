# canonical_huffman_code.py
The implemention for canonical_huffman_code
<<<<<<< HEAD
## Overview
Canonical Huffman coding is an optimized variant of Huffman coding that minimizes header sizes by only storing the code lengths rather than the full code tree. The decoder reconstructs the tree based on the code lengths during decoding, making it more efficient for space-constrained environments.

## How to Use

### Classes:
1. **CanonicalIntHuffmanEncoder**: 
   - Encodes integer symbols in the range `[0, alphabet_size)` using Canonical Huffman coding.
   - Uses only code lengths in the header instead of full trees.

2. **CanonicalIntHuffmanDecoder**: 
   - Decodes the encoded data by reconstructing the Huffman tree from the stored code lengths.

### Example Usage:

```python
from canonical_huffman_code import CanonicalIntHuffmanEncoder, CanonicalIntHuffmanDecoder
from scl.core.data_block import DataBlock

# Initialize encoder and decoder
encoder = CanonicalIntHuffmanEncoder(alphabet_size=256)
decoder = CanonicalIntHuffmanDecoder(alphabet_size=256)

# Sample data
data = [1, 2, 3, 1, 2, 3, 1, 2, 3]
data_block = DataBlock(data)

# Encoding the data
encoded = encoder.encode_block(data_block)

# Decoding the data
decoded, _ = decoder.decode_block(encoded)

# Check if the decoded data matches the original
assert decoded.data_list == data
=======
>>>>>>> 76e716b5e553c1a5ec4d50353d6054d86dde3239

# lz77_canonical_benchmark.py
The test file for baseline compared with canonical_huffman_code

# How to run the code
Go to compressors folder and run the following commend
<<<<<<< HEAD
```

## Canonical Huffman Coding - Unit Tests

### 1. Test with Empty Block
This test ensures that encoding and decoding an empty block works correctly. It checks that the decoder returns an empty list and verifies that the number of bits consumed matches the expected header size.

### 2. Test with Single Symbol
This test ensures that encoding and decoding works properly when only one symbol is present in the input. It verifies that the decoder correctly returns the original data.

### 3. Test with Two Symbols
This test verifies that the encoder and decoder handle cases where exactly two symbols are used, one more frequent than the other. It checks the correctness of the encoded and decoded output.

# lz77_canonical_benchmark.py

## run file
```shell
python lz77_canonical_benchmark.py -i ../testfiles/big.txt
``` 

## run folder
```shell
python lz77_canonical_benchmark.py --data-folder ../testfiles/data/
```
=======
```shell
python lz77_canonical_benchmark.py -i ../testfiles/big.txt
``` 
