## Entropy Coding Improvements for LZ77 in SCL

**Authors**: Jiayu Chang, Boyu Han, Yisi Lyu  
**Affiliation**: Stanford University  
**Code repository**: `https://github.com/Turquoise-T/Stanford_EE274_project`

---

## 1 Introduction

We aim to improve the entropy-coding stage of the Stanford Compression Library (SCL) LZ77 compressor, as illustrated in Figure 1. SCL currently encodes literals, match lengths, and offsets using a single empirical Huffman scheme, whereas production codecs (e.g., gzip, zstd) employ more advanced methods such as canonical Huffman coding or table-based Asymmetric Numeral Systems (tANS/FSE). We will replace the SCL entropy encoder with these modern methods, keeping the LZ77 parser unchanged, and evaluate their impact on compression ratio, speed, and header overhead. This will quantify how much benefit modern entropy coders provide when the LZ77 parsing is held fixed.

This problem is interesting because it bridges the gap between the theoretical concepts taught in class and the practical entropy-coding techniques deployed in production compression systems. Within a single LZ77 framework, we can systematically study trade-offs in compression ratio, runtime, and header overhead between empirical Huffman, canonical Huffman, and tANS-based coders.

Figure 1: High-level LZ77 block diagram (parser + entropy coding stage).

---

## 2 Literature and code review

We will study the LZ77 entropy-coding framework in the Stanford Data Compression Notes Tatwawadi (2025a) and the existing SCL implementation (`lz77.py` and `lz77_sliding_window.py`) Tatwawadi (2025b). For canonical Huffman coding, we draw on standard references and applications in full-text indexing and image compression Zhang et al. (2008); Khaitu and Panday (2018). For tANS, we follow the ANS/tANS theory and Finite State Entropy design described in Duda (2013); Gibbons (2019); Collet (2013a,b); Tatwawadi (2018). As a stretch goal, we may also consult work on LZMA-style context modeling with range coding Tao (2024); Wikipedia contributors (2025).

All code for this project is currently hosted at `https://github.com/Turquoise-T/Stanford_EE274_project`.

### 2.1 Canonical Huffman coding

Canonical Huffman coding is a restricted form of Huffman coding in which codewords are assigned in a standardized order so that the entire codebook can be reconstructed from just the symbol ordering and their codeword lengths. Instead of transmitting the full tree structure, the encoder sends only a list of code lengths, and the decoder rebuilds the codebook by assigning codes in increasing order of length and then lexicographic order within each length class. This removes the need to store explicit tree pointers and enables very compact model headers and fast table-based decoding.

Zhang et al. introduce canonical Huffman codes in the context of compressed full-text indexing via wavelet trees Zhang et al. (2008). Their canonical Huffman wavelet tree eliminates the memory needed to represent the shape of the Huffman tree, which becomes significant for large alphabets (e.g., 16–32 bit symbols). By encoding only the code lengths and using a canonical layout, their full-text index (CHI) achieves about 1.5× the text size, outperforming both a standard Huffman-based index and a plain wavelet-tree index in space usage, while also simplifying rank/select operations.

Khaitu and Panday study canonical Huffman coding for fractal-based image compression Khaitu and Panday (2018). In their system, canonical codes replace standard Huffman codes in the entropy-coding stage after DCT and fractal modeling. Because the canonical codebook is reconstructed from a compact description, their implementation reduces the overhead associated with transmitting or reconstructing the Huffman tree. They report improved compression ratios and faster compression compared to standard Huffman coding on several color and grayscale images.

Our canonical integer Huffman coder follows the same high-level principle as these works: we first build a standard Huffman code from empirical symbol counts, then convert it to canonical form and transmit only the code lengths. In our case the alphabet consists of LZ77 literals and integer sequence parameters (literal counts, match lengths, and offsets), and the header stores a fixed-size array of code lengths encoded with Elias–Delta codes. We then plug this canonical coder into the existing SCL LZ77 implementation, allowing a direct empirical comparison between empirical Huffman headers and canonical Huffman headers within the same LZ77 parsing and probability model.

### 2.2 Table-based Asymmetric Numeral Systems (tANS)

For the tANS component, we combine the theoretical foundations of Asymmetric Numeral Systems with implementation patterns from production compressors. On the theory side, we rely primarily on Duda’s work on ANS Duda (2013), which formalizes ANS as a single-state alternative to arithmetic coding and shows how near-Shannon performance can be achieved using integer-only renormalization. Gibbons’ tutorial “Coding with Asymmetric Numeral Systems” Gibbons (2019) complements this by giving a step-by-step derivation of the encoder/decoder recurrences and clarifying how to move from symbol probabilities and counts to normalized frequency tables and state transitions.

For concrete tANS mechanics and table construction, our main reference is Yann Collet’s Finite State Entropy (FSE) design Collet (2013a,b). FSE explains how to normalize empirical symbol counts to a power-of-two table size, distribute symbols across the state space, and derive the decoding and encoding tables that drive the ANS state machine. In our project, we follow this workflow in Python: for each block of match lengths and offsets, we estimate counts, normalize them to an ANS table size, build the corresponding tANS tables, and serialize a compact description of the normalized frequencies in the LZ77 block header so that the decoder can reconstruct the identical state machine.

To understand how tANS is used in real LZ77-style compressors, we draw on the Zstandard format (RFC 8878) and reference implementation Collet and Kucherawy (2021); Collet et al. (2016), which use FSE tables for match lengths and offsets while literals are typically Huffman-coded, and define a clear binary layout for transmitting table descriptions. Apple’s LZFSE reference implementation provides a second example of a Lempel–Ziv compressor whose entropy stage is based on finite-state entropy Apple Inc. (2016). Finally, we use Tatwawadi’s ANS exposition Tatwawadi (2018) to keep our notation and intuition aligned with the Stanford compression notes. Taken together, these sources give us (i) the ANS theory we need, (ii) a concrete tANS algorithm and table format to implement, and (iii) tested integration patterns for plugging a tANS-based entropy stage into an LZ77 compressor in SCL.

---

## 3 Methods

We will keep the SCL LZ77 parser unchanged and replace only the entropy-coding layer.

**Canonical Huffman coding.** We will implement a canonical Huffman coder for literals (and optionally also for the length/offset alphabets), following the Deflate/gzip design. The encoder will transmit only code lengths so that the decoder can reconstruct the codebook deterministically, reducing header bytes and improving decode speed. We will integrate this coder into the entropy layer through SCL’s `DataEncoder`/`DataDecoder` interface.

**tANS / Finite State Entropy.** We will implement tANS for match lengths and offsets (and literals or literal runs if present), replacing the current binned empirical-Huffman path. For each block, we will normalize empirical counts to a power-of-two table size, write the normalized counts and/or table description in the block header, and use precomputed state-transition tables for encoding and decoding, while preserving SCL’s logarithmic binning (Zstd-style). The implementation will follow published FSE algorithms and be adapted to SCL.

We will compare our implementations against SCL’s baseline empirical Huffman encoder on the Squash Corpus Nemerson (2015) and the Canterbury Corpus University of Canterbury (1997). For quantitative evaluation, we will measure compression ratio (including headers), encoding speed, and decoding speed, and report per-file tables with 95% confidence intervals and p-values (paired t-test or Wilcoxon signed-rank test, as appropriate). For qualitative analysis, we expect canonical Huffman to reduce header bytes and improve decode speed, and tANS to improve compression ratio for the length/offset streams; we will inspect literal distributions and match-parameter statistics to explain observed differences. We will also track header overhead explicitly and present summary tables and compression-ratio/speed plots.

---

## 4 Progress report

### 4.1 Canonical Huffman coding

For the canonical Huffman coding component, we have implemented a standalone integer canonical Huffman coder in `canonical_huffman_code.py`. It provides `CanonicalIntHuffmanEncoder` and `CanonicalIntHuffmanDecoder` following the existing `DataEncoder`/`DataDecoder` interface and can be plugged into the LZ77 entropy-coding layer in place of the empirical Huffman coder.

Next, we plan to construct simple synthetic test files and regression tests to further validate the correctness and robustness of the canonical coder, and then run systematic experiments on standard datasets (e.g., the Squash and Canterbury corpora) to compare canonical Huffman coding against the existing empirical Huffman coder in terms of header size, overall compression ratio, and runtime.

### 4.2 tANS

For the tANS entropy-coding component, we have implemented a complete LZ77 + tANS pipeline by adding four modular files: `tans_core.py` (tANS state machine construction and encode/decode), `tans_int_coder.py` (tANS for integer LZ77 streams), `tans_byte_coder.py` (tANS for literal-byte streams), and `lz77_streams_tans.py` (the interface layer that connects tANS to the existing LZ77 framework). These modules conform to the original `DataEncoder`/`DataDecoder` abstraction, allowing tANS to be plugged into the entropy-coding stage without modifying the LZ77 parsing or decoding logic.

In the next steps, we plan to refine the tANS table construction process (e.g., normalized-frequency-based table reconstruction, FSE-style symbol spreading, and cross-block table reuse) to further reduce metadata overhead and improve compression performance. We will then conduct systematic experiments on standard datasets to compare the tANS backend against the existing Huffman coder in terms of header size, overall compression ratio, and runtime efficiency.

Looking ahead from the Nov 20 milestone, our plan for the remaining weeks is as follows. For the rest of Week 9 (Nov 17–23), we will (i) finish unit tests for both the canonical Huffman coder and the tANS modules, (ii) clean up the APIs, and (iii) run small sanity experiments on synthetic data and a few Squash/Canterbury files to validate correctness and stability. In addition, we plan to deepen our understanding of tANS, since the current implementation is a simplified version, and perform bug fixing and implementation refinements to improve the accuracy and robustness of the entropy-coding backend.

In Week 10 (Nov 24–30), we will run the full experimental sweep on the Squash and Canterbury corpora, including per-file compression ratio (with headers), encode/decode speed, and header-size breakdowns for the baseline, canonical Huffman, and tANS backends, and iterate on tANS table-construction parameters if we see obvious issues. In Week 11 (Dec 1–4), we will finalize the experiments, generate the main plots and tables, write up the analysis sections of the report, and prepare the final slides, keeping a small buffer for any last bug fixes or reruns that come up during writing.

---

## References

Apple Inc. 2016. LZFSE Compression Library and Command Line Tool. GitHub repository. `https://github.com/lzfse/lzfse`. Accessed: 2025-11-20.

Yann Collet. 2013a. Finite State Entropy: A New Breed of Entropy Coder. Blog post. `https://fastcompression.blogspot.com/2013/12/finite-state-entropy-new-breed-of.html`. Accessed: 2025-11-20.

Yann Collet. 2013b. FiniteStateEntropy. GitHub repository. `https://github.com/Cyan4973/FiniteStateEntropy`. Accessed: 2025-11-20.

Yann Collet et al. 2016. Zstandard – Fast Real-Time Compression Algorithm. GitHub repository. `https://github.com/facebook/zstd`. Reference implementation, accessed: 2025-11-20.

Yann Collet and Murray Kucherawy. 2021. Zstandard Compression and the application/zstd Media Type. RFC 8878. doi:10.17487/RFC8878.

Jarek Duda. 2013. Asymmetric Numeral Systems: Entropy Coding Combining Speed of Huffman Coding with Compression Rate of Arithmetic Coding. arXiv preprint (2013). arXiv:1311.2540, `https://arxiv.org/abs/1311.2540`.

Jeremy Gibbons. 2019. Coding with Asymmetric Numeral Systems. In Mathematics of Program Construction (Lecture Notes in Computer Science, Vol. 11825). Springer, 444–465. doi:10.1007/978-3-030-33636-3_16.

Shree Ram Khaitu and Sanjeeb Prasad Panday. 2018. Canonical Huffman Coding for Image Compression. In 2018 IEEE 3rd International Conference on Computing, Communication and Security (ICCCS). IEEE, 183–189.

Evan Nemerson. 2015. Squash Test Corpus. GitHub repository. `https://github.com/nemequ/squash-corpus`. Accessed: 2025-11-20.

Nigel Tao. 2024. How XZ and LZMA Work: Part 1 – Range Coding. Blog post. `https://nigeltao.github.io/blog/2024/xz-lzma-part-1-range-coding.html`. Accessed: 2025-11-20.

Kedar Tatwawadi. 2018. Understanding the ANS Compressor. Blog post. `https://kedartatwawadi.github.io/post--ANS/`. Accessed: 2025-11-20.

Kedar Tatwawadi. 2025a. Lossless Compression: LZ77 and Entropy Coding. Stanford Data Compression Notes. `https://stanforddatacompressionclass.github.io/notes/lossless_iid/lz77.html`. Accessed: 2025-11-20.

Kedar Tatwawadi. 2025b. Stanford Compression Library (SCL). GitHub repository. `https://github.com/kedartatwawadi/stanford_compression_library`. Accessed: 2025-11-20.

University of Canterbury. 1997. The Canterbury Corpus. Corpus collection. `https://corpus.canterbury.ac.nz/descriptions/#cantrbry`. Accessed: 2025-11-20.

Wikipedia contributors. 2025. Lempel–Ziv–Markov Chain Algorithm. Wikipedia, The Free Encyclopedia. `https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm`. Accessed: 2025-11-20.

Yi Zhang, Zhili Pei, Jinhui Yang, and Yanchun Liang. 2008. Canonical Huffman Code Based Full-Text Index. Progress in Natural Science 18, 3 (2008), 325–330. doi:10.1016/j.pnsc.2007.11.001.


