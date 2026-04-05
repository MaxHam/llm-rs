# nanogpt-ish
This repository will be a place for me to learn language model basics. Target is to write a small gpt-style model that can run on CPU, written in candle(similar to python torch). Its purely educational and for fun.

The code is heavily influenced by the ["Let's build GPT: from scratch, in code, spelled out."](https://github.com/karpathy/ng-video-lecture) lecture by Andrej Karpathy.
Its called "nanogpt-ish", because it follows Karpathys structure, but is most likely much worse implemented.

## Requirements
- Rust

## How to use
Run 
```shell
cargo run --release # for optimal compilation and fast training
# with cuda
cargo run --release --features cuda
```

## Roadmap
- [x] Byte Pair Encoding
- [x] Tokenizer 
- [ ] Tokenize a big corpus
- [x] Dense Architecture LLM 
- [ ] Mixture of Expert Architecture