# Multi-Horizon Forecasting for Limit Order Books (MHF)
**Authors:** Zihao Zhang, Stefan Zohren (Oxford-Man Institute)

## Core Concept
This paper proposes a "Super Model" architecture that combines the strengths of advanced feature extraction (DeepLOB) with the temporal consistency of sequence-to-sequence generation.

### The Problem
*   Standard models make single-point predictions or "one-shot" vector predictions.
*   LOB data is stochastic and has a low signal-to-noise ratio.
*   We need to predict a *path* (multi-horizon) to make better trading decisions.

## The "Super Model" Architecture (Section 4)
The paper explicitly combines these two components:

### 1. The Encoder: DeepLOB (Figure 4)
*   **Source:** Adapted from the original DeepLOB (Zhang et al., 2019a).
*   **Structure (Detailed on Page 6):**
    *   **Convolutional Block:** Uses 1D CNNs to extract features from specific order book levels (prices/volumes), automating feature extraction from raw LOB data. It acts like an auto-regressive model in the time dimension and handles the low signal-to-noise ratio via parameter sharing.
    *   **Inception Module:** Takes the feature maps from the Conv block and applies multiple parallel convolutions with different kernel sizes. This captures interactions over different time horizons (short vs long term) simultaneously.
    *   **LSTM Encoder:** A final LSTM layer captures longer-term temporal behavior from the Inception features and produces the final context vector connectivity to the decoder.
*   **Role:** Compresses the last 100 timesteps of raw LOB data into a meaningful "Context Vector".
*   **(Visual Reference):** Figure 3 illustrates the Attention mechanism, and Figure 4 shows the full DeepLOB + Encoder-Decoder pipeline.

### 2. The Decoder: Seq2Seq + Attention
Instead of a simple linear output, they use a generative decoder.

#### Option A: Basic Seq2Seq (Eq 1-3)
*   Uses an LSTM/RNN to unroll predictions one step at a time.
*   **Bridge:** The final hidden state of the Encoder becomes the initial state of the Decoder.
*   **Autoregressive:** The prediction at $t$ is fed as input to $t+1$.

#### Option B: Attention Mechanism (Eq 4-7)
*   **Problem:** A single context vector struggles to hold all information for long sequences.
*   **Solution:** Uses **Luong Attention**.
*   **Mechanism:**
    *   The decoder looks back at *all* encoder hidden states (not just the last one).
    *   It calculates "Energy" scores to decide which part of the history is most relevant for predicting the *current* step $t$.
    *   Computes a weighted sum (Context Vector $c_t$) to guide the prediction.

## Hardware Acceleration (IPU)
The paper also highlights that these recurrent (Seq2Seq) models are slow on GPUs due to their sequential nature. They use **IPUs (Intelligence Processing Units)** from Graphcore, which allow for massive parallelization of the fine-grained operations, achieving significant speedups over GPUs.
