# ASR leaderboard evaluation report

## Executive summary

I reproduced part of the [Hugging Face Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) with four English Whisper checkpoints and Wav2Vec2 Base. I evaluated all 2,620 LibriSpeech `test-clean` recordings, then used a 300-recording Earnings22 sample to test whether the result transferred from clean audiobook speech to conversational financial calls.

On the saved full LibriSpeech run, **Whisper Small has the lowest WER (2.84%)**. **Wav2Vec2 is the efficient choice**: 3.28% WER at more than 100× real time on this Mac. That advantage does not transfer to Earnings22, where Wav2Vec2 reaches 42.01% WER while Small and Medium reach 11.45% and 11.06%. Small is the best observed accuracy–cost compromise across these two datasets, but this is a conditional recommendation rather than a claim of universal superiority.


On the saved full LibriSpeech run, Whisper Small has the lowest WER (2.84%), but this ranking against Medium (3.30%) reverses once timestamp-induced repetitions are removed (2.79% vs 2.66%).

## 1. Evaluation design

The original LibriSpeech experiment used 300 recordings selected speaker-first, then balanced across duration thirds. It covered all 40 speakers. Against 1,000 random samples of the same size, its words-per-second distribution was closer to the full set than 97.2% of random draws, and its duration-group balance was better than 85.2%. However, its continuous duration distribution was worse than 72.5% of random draws. The design controlled the features it targeted; it was not better on every measure.

The sample showed that Base outperformed Tiny and Small outperformed Base. It could not reliably distinguish Small from Medium. Its 95% confidence intervals also missed the later full-test WER for Tiny and Small, so I use the complete LibriSpeech run for the headline results.

I added Earnings22 because LibriSpeech is unusually clean and Wav2Vec2 was fine-tuned on it. The sample covers all six source calls and approximately preserves their proportions and the full split's duration, word-count, and speaking-rate distributions. Earnings22 supplies call IDs, not reliable speaker IDs, so uncertainty is grouped by call. With only six calls, those intervals are broad.

## 2. Accuracy and efficiency

Primary WER uses the Whisper English normalizer loaded from the Whisper tokenizer, including its British-to-American spelling map. Confidence intervals resample whole speakers.

| Model | Local LibriSpeech full-test WER | 95% confidence interval | Published LibriSpeech WER | Throughput | MPS memory |
|---|---:|---:|---:|---:|---:|
| `tiny.en` | 5.87% | 4.64–7.67% | 5.66% | 28.8× | 1,114 MB |
| `base.en` | 3.84% | 3.37–4.34% | 4.25% | 16.5× | 1,198 MB |
| `wav2vec2-base` | 3.28% | 2.91–3.65% | 3.40% | **>100×** | **1,089 MB** |
| `small.en` | **2.84%** | 2.45–3.26% | 3.05% | 6.4× | 2,442 MB |
| `medium.en` | 3.30% | 2.33–4.60% | 3.02% | 2.0× | 5,192 MB |

Small improves on Wav2Vec2 by 0.43 percentage points (paired 95% interval: 0.04–0.84). The Small–Medium interval includes zero. Local values are within 0.41 points of the archived published results; remaining differences can reflect model/library revisions and decoding settings that the older artifacts do not fully record.

Published sources: [`tiny.en`](https://huggingface.co/openai/whisper-tiny.en/discussions/27/files), [`base.en`](https://huggingface.co/openai/whisper-base.en/discussions/18/files), [`small.en`](https://huggingface.co/openai/whisper-small.en/discussions/17/files), [`medium.en`](https://huggingface.co/openai/whisper-medium.en/discussions/19/files), and [`wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h). The Wav2Vec2 source reports 3.4% to one decimal place.

The runtime benchmark used the same 30 recordings, three repeats, and one untimed warm-up recording so one-time MPS setup did not count as latency. Wav2Vec2's repeat-level throughput was 111×, 140×, and 139×, so I report it as **>100×** rather than a falsely precise average. The Whisper benchmark requested timestamps while Wav2Vec2 did not, so cross-architecture timing is approximate. MPS memory is Apple's driver-allocation reading after warm-up, not peak system memory.

## 3. Normalization

Raw WER is not comparable across these systems: LibriSpeech references are uppercase and unpunctuated, Whisper emits formatted text, and Wav2Vec2 emits mostly unformatted text. I apply each normalizer equally to references and predictions.

| Model | Raw WER | Punctuation removed | Custom rules | Whisper English |
|---|---:|---:|---:|---:|
| `tiny.en` | 99.86% | 6.53% | 6.08% | 5.87% |
| `base.en` | 98.96% | 4.49% | 4.05% | 3.84% |
| `wav2vec2-base` | 3.39% | 3.43% | 3.36% | 3.28% |
| `small.en` | 98.80% | 3.41% | 3.05% | 2.84% |
| `medium.en` | 99.43% | 3.82% | 3.49% | 3.30% |

The custom rules handle digits, currency, percentages, ordinals, contractions, titles, hyphens, spelling variants, and selected fillers. For example, `$2.23` and “two dollars and twenty three cents” now normalize identically. Small remains the lowest-WER model under both the custom and Whisper English rules.

The strongest normalization example is Earnings22 Wav2Vec2: **104.83% raw WER becomes 42.01% after common normalization**. Raw WER can exceed 100% when insertions outnumber the reference words.

## 4. Out-of-domain result: Earnings22

| Model | Local n=300 WER | 95% interval, resampling calls | Published Earnings22 WER |
|---|---:|---:|---:|
| `tiny.en` | 23.24% | 15.27–34.10% | 19.12% |
| `base.en` | 12.83% | 10.20–16.03% | 15.09% |
| `wav2vec2-base` | 42.01% | 38.50–48.08% | 48.47% |
| `small.en` | 11.45% | 9.41–13.76% | 12.97% |
| `medium.en` | **11.06%** | 9.04–14.10% | **12.63%** |

Wav2Vec2 is worst overall and on five of six calls; on call 4485244, Tiny is worse (42.77% versus 39.20%). The 0.39-point Medium–Small difference is unresolved across only six calls. The local sample reproduces the published model ordering, although the published values use the complete split and may use different decoding settings.

This run also exposed a configuration failure. Forcing timestamp generation on every short Whisper clip produced repeated continuations and sharply inflated WER. Using ordinary short-form decoding for clips up to 30 seconds reduced WER by 6–24 points, depending on the model.

Listening found that some references disagree with audible speech, retain fillers that models omit, or use written forms where a model emits the spoken equivalent. These examples came from an error-enriched queue, so they show possible failure modes rather than their dataset-wide frequency. A high-stakes choice between Small and Medium would benefit from a random, blinded reference audit.

## 5. What WER misses

WER counts incorrect words but does not distinguish scattered mistakes from a missing or invented phrase. I therefore add a **severe-output rate**: the percentage of recordings containing either a consecutive omission or a consecutive insertion of at least three words. The common three-word threshold makes the rule easy to interpret and treats both failure directions consistently.

| Model | Severe-output recordings | Severe-output rate | ≥3-word consecutive omissions | Longest consecutive omission | ≥3-word consecutive insertions | Longest consecutive insertion |
|---|---:|---:|---:|---:|---:|---:|
| `tiny.en` | 8 | 0.31% | 1 | 3 words | 7 | 392 words |
| `base.en` | 3 | 0.11% | 0 | 2 words | 3 | 4 words |
| `wav2vec2-base` | 1 | **0.04%** | 0 | 2 words | 1 | 3 words |
| `small.en` | 4 | 0.15% | 1 | 21 words | 3 | 12 words |
| `medium.en` | 18 | 0.69% | 14 | 10 words | 4 | 220 words |

These are the saved full-run outputs. They also exposed a decoding issue: I re-decoded the union of 30 flagged Whisper recordings, disabling timestamps for 28 clips at or below 30 seconds; Whisper requires timestamps for the two longer clips. Tiny's 392-word repetition, Small's 12-word repetition and 21-word omission, and Medium's 220-, 70-, and 64-word repetitions disappeared. The resulting hybrid severe counts were 7, 1, 2, and 16 for Tiny, Base, Small, and Medium. This is not a replacement full-run score because the clips were selected from the original failures. It shows that several extreme repetitions were configuration artifacts and that a future full rerun should use consistent short-form decoding.

Medium's speaker-908 omissions did persist. Across that speaker's 57 full-test recordings, Small has 5.18% WER and Medium 14.17%; listening confirmed that the sampled omitted openings are audible and correctly referenced.

For a captioning product owner, the severe-output rate complements average WER by flagging phrase-level failures that may require human review. The threshold is an exploratory operational choice and should be validated before becoming a release criterion.

## 6. Data characterization and reliability

The full 2,620-recording slice analysis shows that slower speech is harder for every model. Small scores 3.82% WER on the slow third versus 2.45% on the fast third; Medium scores 5.15% versus 2.55%. The conclusion remains under broader 25%/75% cutoffs: every model still has higher WER in the slow group. Short and low-word-count clips are also generally harder, partly because one error is a larger fraction of a short reference and there is less context.

I distinguish integrity checks from evidence that derived metadata are stable enough for slicing:

| Check type | Test | Result | Meaning |
|---|---|---:|---|
| Integrity | Saved versus freshly decoded duration | 0.000000 s maximum difference | Rows and audio files are joined correctly |
| Integrity | Speaker field versus sample-ID prefix | 100% agreement | Speaker clustering is implemented correctly |
| Text-feature stability | Raw versus normalized word-count rank | 0.9987 | Relative utterance length barely changes |
| Text-feature stability | Speaking-rate group agreement | 91.3% | Broad rate slices are not sensitive to tokenization |
| Method agreement | Two leading-silence thresholds | 0.8475 rank correlation; 10 ms median difference | Suitable for broad diagnostics, not exact boundaries |
| Slice stability | Slow versus fast under 25%/75% cutoffs | Slow is worse for all five models | The main conclusion survives different cut points |

## 7. Limitations

- Earnings22 uses 300 segments from only six source calls. It establishes a strong domain shift but cannot precisely represent financial-call speech in general.
- The historical leaderboard artifacts do not preserve every model revision and decoding setting, so the published comparisons are not execution-identical reproductions.
- The reviewed examples were selected for large errors and cannot estimate how common reference problems are.
- The targeted no-timestamp check is diagnostic; a complete consistently decoded Whisper rerun would be needed to replace the headline scores.
- Runtime uses 30 recordings and three repeats on one Mac; memory is the MPS driver allocation after warm-up, not peak total process memory.
- The severe-output threshold was motivated by observed failures and should be tested on another dataset before becoming a release criterion.

## 8. Automation

The project already has fixed manifests, resumable inference, cached predictions, and a saved configuration. For regular use, I would add scheduled full runs, permanent result storage, and checks for missing/duplicate rows or unexpected changes in accuracy, severe outputs, and runtime.

## 9. AI-use disclosure

I used ChatGPT to help write and debug code and edit the notebooks, README, and report. I chose and directed the analyses, ran and reviewed the experiments, and made the final interpretations. AI tools did not generate the model predictions, reference transcripts, timing measurements, or listening judgments.