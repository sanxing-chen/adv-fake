# AdvFake: Adversarial Fake News Generation

This repository contains code for the paper:

**[Real-time Fake News from Adversarial Feedback](https://arxiv.org/abs/2410.14651)**  
Sanxing Chen, Yukun Huang, Bhuwan Dhingra

If you find this work useful in your research, please cite:

```
@misc{chen2024advfake,
      title={Real-time Fake News from Adversarial Feedback}, 
      author={Sanxing Chen and Yukun Huang and Bhuwan Dhingra},
      year={2024},
      eprint={2410.14651},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.14651}, 
}
```

> [!CAUTION]
> DISCLAIMER: This code is for research purposes only, specifically for studying misinformation detection
> and improving fact-checking systems. Any use of this code for generating and spreading actual 
> misinformation is strictly prohibited and unethical.

> [!IMPORTANT]  
> RESPONSIBLE USAGE:
> - Use only for academic research and improving detection systems
> - Do not deploy for generating actual fake news
> - Follow ethical guidelines for AI research

## Overview

The system works by:
1. Taking real news articles as input
2. Generating fake versions through various rewriting strategies
3. Evaluating plausibility using LLM-based scoring
4. Using retrieval-augmented generation (RAG) to improve fake news detection
5. Iteratively improving the fake news generation based on detection feedback

## Data

![License](https://img.shields.io/badge/license-ODC--BY-brightgreen)


Our final dataset with human validation is available on Hugging Face:
- [AdvFake Dataset](https://huggingface.co/datasets/sanxing/advfake)
- Each row contains the original true news, the corresponding generated fake news in the final round.

We also provide our DPR-based retrieval dataset on Hugging Face:
- [AdvFake News-DPR](https://huggingface.co/datasets/sanxing/advfake_news_please)
- This dataset is sourced from [Common Crawl](https://commoncrawl.org/news-crawl) using [news-please](https://github.com/fhamborg/news-please)
- The date range is from 2024-03-01 to 2024-03-14

These datasets are under ODC-BY license. You are free to share, adapt, and use them with attribution.

## Usage

You need to set up the environment variables for OpenAI or Gemini API keys. If you want to use Google Search, you need to set up the `SERPAPI_API_KEY` environment variable.

1. Generate the initial fake news dataset
```
python generate.py --target init --first-round
```

*You can use `--preflight` to check the pipeline with a few examples.*

In this first step, the program will download both the datasets from Hugging Face and generate the initial fake news dataset. Some extra time is needed for indexing the DPR corpus, which will be saved in the local directory. A GPU server is recommended for this step. As a reference, it takes about 10 minutes to index the DPR corpus with 812k news articles in a single A6000 GPU.

2. Generate the adversarial fake news dataset with RAG-based rationale as feedback
```
python generate.py --source init_round1 --target adv_rag --generation-context-type rag_rationale  --num-rounds 6
```


## Features

- Multiple fake news generation strategies:
  - Entity substitution (replacing names/locations)
  - Open-ended rewriting
  - RAG-enhanced generation
  - Rationale-guided adversarial generation

- Retrieval capabilities:
  - DPR (Dense Passage Retrieval) for finding related articles
  - Google Search integration
  - Context-aware generation and evaluation

