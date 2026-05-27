# AI Classification of RF Signals

Multi-label classification of overlapping wireless signals from impaired RF captures.

## Requirements

MATLAB R2021b or newer, with the **Deep Learning** and **Signal Processing** toolboxes.
Experiment 1 also needs the **Communications** and **Parallel Computing** toolboxes;
Experiment 2 also needs the **Image Processing** toolbox.

## Experiment 1 — synthetic signals

Generates synthetic OFDM signals and classifies WLAN / LTE / 5G by their
cyclic-prefix structure. No external data required.

```
matlab -batch "run_full_experiment"
```

## Experiment 2 — real RFSS dataset

Multi-label classification of overlapping GSM / UMTS / LTE / 5G NR signals from
the real RFSS dataset.

Download the data file (1.4 GB) into `./data` first:

```
mkdir -p data
curl -L -o data/rfss_single.h5 \
  "https://huggingface.co/datasets/Chrishao/rfss/resolve/main/data/rfss_single.h5"
```

Then run:

```
matlab -batch "run_rfss_experiment"
```

Dataset source: <https://huggingface.co/datasets/Chrishao/rfss>
