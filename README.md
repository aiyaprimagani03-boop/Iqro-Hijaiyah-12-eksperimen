# Iqro' Volume 1 hijaiyah letter recognition: 12 experiments

Code and results for the study *A Study of Deep Learning for the Speech Recognition of Consonant–Vowel Readings in the Iqro' Method*. Four pretrained speech models (wav2vec 2.0, HuBERT, WavLM, Whisper-tiny) are fine-tuned under three input scenarios, giving 12 experiments:

| Scenario | Input |
|---|---|
| Base | backbone only |
| +MFCC | backbone fused with 20 MFCC |
| +log-Mel | backbone fused with 80-band log-Mel |

## Repository structure
```
Iqro-Hijaiyah-12-eksperimen/
├── IQRO_Hijaiyah_Speaker_12Eksperimen.ipynb   # full pipeline: preprocessing, features, training, evaluation (Google Colab)
├── hasil/
│   ├── split_manifest.csv                     # assignment of every recording to train / validation / test
│   ├── ringkasan_12_eksperimen.csv            # summary metrics of the 12 experiments
│   ├── uji_mcnemar.csv                        # paired McNemar tests with Holm correction
│   ├── Test Predictions/                      # test_predictions.csv of each experiment (<model>/Skenario 1-3/)
│   └── Output Training and Confusion Matrix/  # training curves and confusion matrices
├── analisis/
│   ├── analysis.py                            # recomputes PER, Wilson intervals, McNemar, bootstrap, macro F1
│   └── make_cm.py                             # confusion matrices in Iqro' order with Arabic letters
└── README.md
```

## Dataset
860 utterances from 10 adult speakers (5 male / 5 female, aged 19–23), each reading the 29 hijaiyah letters with fathah of Iqro' Volume 1 three times (870 planned; 10 removed because of background noise). Each recording is one consonant–vowel syllable, such as "ba", and is labelled with one of 29 classes.
Recordings were made in an ordinary sound-insulated room with Android smartphones and wired earphone microphones held 10 cm from the mouth: speaker S06 used an Android 13 phone (44.1 kHz) with a macaron-type earphone, and the other speakers used an Android 15 phone (48 kHz) with a Log-On LO-HF630 earphone. All files were converted to 16 kHz mono and segmented per letter in Audacity.
The split is speaker-independent (GroupShuffleSplit, seed 42): 600 training recordings (7 speakers), 87 validation recordings (S03), and 173 test recordings (S02, S09).
The audio recordings are **not included** in this repository; they are available from the corresponding author upon reasonable request.

## How to run
1. Open `IQRO_Hijaiyah_Speaker_12Eksperimen.ipynb` in Google Colab (GPU runtime; experiments were run on an NVIDIA L4).
2. Set the dataset and output paths in the configuration cell.
3. Run all cells in order. All experiments use seed 42 and were run once.

Software: Python 3.13.16, PyTorch 2.14.1 (CUDA 13.0), Transformers 5.22.1, librosa 0.11.0.
Checkpoints: `facebook/wav2vec2-base`, `facebook/hubert-base-ls960`, `microsoft/wavlm-base-plus`, `openai/whisper-tiny`.

## Results (test set: 173 utterances from 2 unseen speakers)
| Backbone | Input | Correct (of 173) | Accuracy (%) | WER (%) | CER (%) | PER (%) | Invalid (%) | Time (ms) |
|---|---|---|---|---|---|---|---|---|
| wav2vec 2.0 | Base | 0 | 0.00 | 100.00 | 57.89 | 68.26 | 100.00 | 2.35 |
| wav2vec 2.0 | +MFCC | 118 | 68.21 | 31.79 | 16.03 | 14.67 | 11.56 | 2.48 |
| wav2vec 2.0 | +log-Mel | 117 | 67.63 | 32.37 | 18.18 | 15.57 | 9.83 | 2.34 |
| HuBERT | Base | 115 | 66.47 | 33.53 | 20.57 | 17.66 | 7.51 | 2.18 |
| HuBERT | +MFCC | 114 | 65.90 | 34.10 | 18.90 | 18.86 | 4.05 | 2.41 |
| HuBERT | +log-Mel | 115 | 66.47 | 33.53 | 17.94 | 17.07 | 2.89 | 2.39 |
| WavLM | Base | 88 | 50.87 | 49.13 | 25.36 | 24.85 | 32.37 | 3.57 |
| WavLM | +MFCC | 116 | 67.05 | 32.95 | 19.62 | 17.37 | 9.83 | 3.51 |
| WavLM | +log-Mel | 112 | 64.74 | 35.26 | 19.86 | 18.56 | 9.25 | 3.66 |
| Whisper-tiny | Base | 125 | 72.25 | 27.75 | 17.70 | 13.77 | 0.00 | 5.85 |
| Whisper-tiny | +MFCC | 111 | 64.16 | 35.84 | 24.64 | 19.16 | 0.00 | 6.02 |
| Whisper-tiny | +log-Mel | 101 | 58.38 | 41.62 | 25.36 | 20.06 | 2.31 | 5.79 |

WER equals 1 − accuracy because every label and every prediction is one word. Whisper-tiny Base has the highest accuracy, but it does not differ significantly from wav2vec 2.0 +MFCC (McNemar p = 0.337) or from HuBERT Base (p = 0.164).

**Note on PER.** PER treats each digraph (kh, sy, sh, ts, th, dz, dh, zh, gh) as one phoneme and removes the numeric prefix of the labels, so Alif (01a) and Hamzah (28a) are both written "a". The column `phone_edit_distance` in `test_predictions.csv` and the PER in `summary_comparison.png` were produced with an earlier definition that kept the digits, so they are higher. To reproduce the PER values above:
```
python analisis/analysis.py "hasil/Test Predictions"
```

## Citation
If you use this code, please cite:
> A. P. Z. Afnan, S. Hidayat, "A Study of Deep Learning for the Speech Recognition of Consonant–Vowel Readings in the Iqro' Method," *Turkish Journal of Mathematics and Computer Science*, 2026 (under review).

## License
Code: MIT License. Contact: aiyaprimagani03@students.unnes.ac.id
