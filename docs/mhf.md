# DeepLOB + Seq2Seq "Super Model" Architecture

This architecture represents a hybrid approach to Limit Order Book (LOB) forecasting, designed to solve the specific challenges of high-frequency cryptocurrency trading.

It combines two powerful neural network paradigms:
1.  **DeepLOB Encoder:** A specialized Convolutional Neural Network (CNN) designed to extract "visual" patterns from the shape of the order book.
2.  **Seq2Seq Attention Decoder:** A Recurrent Neural Network (RNN) designed to generate smooth, temporally consistent future price paths.

---

## 1. The Encoder: DeepLOB (Micro-Structure Extraction)

The goal of the encoder is to take a raw, noisy sequence of order book updates and compress it into a sequence of meaningful "feature vectors."

### Input Data
*   **Dimensions:** $(Batch, T=300, Features=20)$
*   **Window:** The model sees the last **300 seconds (5 minutes)** of data.
*   **Features:** At each second, we have 20 values:
    *   Ask Prices (Levels 1-5), Ask Volumes (Levels 1-5)
    *   Bid Prices (Levels 1-5), Bid Volumes (Levels 1-5)
*   **Normalization:** Crucially, inputs are Z-score normalized (mean 0, std 1) so the network sees relative movements, not absolute price levels (which wander arbitrarily).

### A. The Convolutional Blocks (Local Features)
The raw LOB is treated like an image.
*   **Operation:** We apply 1D Convolutions with small kernels (e.g., size 3) to the price/volume levels.
*   **Purpose:** The network learns to detect local micro-structures, such as:
    *   *Volume imbalances:* Is there more volume on the bid side than the ask side at level 1?
    *   *Spread widening:* Is the gap between Ask 1 and Bid 1 increasing?
*   **Parameter Sharing:** By using convolutions, the same "logic" is applied across different parts of the time series, making the model robust and efficient.

### B. The Inception Module (Multi-Scale Features)
Financial markets exhibit patterns at multiple time scales simultaneously.
*   **Mechanism:** The output of the Conv blocks is fed into three parallel convolution layers with different kernel sizes:
    *   **Small Kernel (1x1):** Captures instantaneous reactions.
    *   **Medium Kernel (3x1):** Captures short-term trends (e.g., last 3 seconds).
    *   **Large Kernel (5x1):** Captures slightly longer trends (e.g., last 5 seconds).
*   **Result:** These parallel streams are concatenated, giving the model a "multi-resolution" view of the market state at every timestep.

### C. The Encoder LSTM (Temporal Context)
*   **Operation:** A Bidirectional LSTM processes the sequence of Inception features.
*   **Purpose:** It aggregates the local features into a global context. It understands the "narrative" of the last 5 minutes (e.g., "The price has been trending up, but volume is drying up").
*   **Output:** A sequence of hidden states $H = \{h_1, h_2, ..., h_{300}\}$ representing the history.

---

## 2. The Decoder: Seq2Seq + Attention (Path Generation)

The goal of the decoder is to look at the history ($H$) and hallucinate a realistic future price path for the next 60 seconds (1 minute).

### The Challenge of "One-Shot" Prediction
Standard models (like the original DeepLOB) try to predict the entire future vector (60 points) at once using a simple linear layer.
*   **Problem:** This often produces "jagged" or independent predictions. The prediction for $t+2$ might not align logically with $t+1$.
*   **Solution:** An **Autoregressive Decoder** generates the path one step at a time.

### A. The Attention Mechanism (Focus)
At every step of generation $t$, the decoder needs to know: *"Which part of the past 5 minutes is relevant right now?"*
*   **Mechanism:**
    1.  The decoder takes its current hidden state $d_t$ (what it just predicted).
    2.  It compares $d_t$ against *all* encoder history states $H = \{h_1, ..., h_{300}\}$.
    3.  It calculates a **score** (or probability) for each past timestamp.
    4.  **Context Vector ($c_t$):** It computes a weighted average of the history $H$ based on these scores.
*   **Intuition:** If the market is crashing, the Attention mechanism might focus heavily on the "Inception" features from 2 minutes ago when a similar crash pattern started.

### B. The Autoregressive Loop
The decoder is an LSTM that runs for 60 steps.
1.  **Input at Step $t$:** It receives two things:
    *   **Context Vector ($c_t$)** (The relevant history).
    *   **Previous Prediction ($y_{t-1}$)** (Where are we right now?).
2.  **Output at Step $t$:** It outputs the prediction for the current second $y_t$.
3.  **Update:** The internal state is updated, and the process repeats for $t+1$.

### C. Teacher Forcing (Training Trick)
*   **During Training:** We feed the *true* previous price $y_{t-1}^{true}$ into the decoder. This helps the model learn faster (it doesn't get derailed by its own early mistakes).
*   **During Inference/Testing:** We feed the *model's own* predicted price $y_{t-1}^{pred}$ into the next step. This is how it will run in the real world.

---

## 3. Comparison: Standard DeepLOB vs. MHF (Super Model)

The key difference lies in **how** they generate the future prediction. The features (Encoder) are identical, but the output mechanism (Decoder) is fundamentally different.

| Feature | Standard DeepLOB | MHF "Super Model" |
| :--- | :--- | :--- |
| **Encoder** | **Deep CNN + Inception + LSTM** (Identical) | **Deep CNN + Inception + LSTM** (Identical) |
| **Output Type** | **One-Shot Vector** | **Autoregressive Sequence** |
| **Prediction Logic** | Predicts all 60 points at once: $Y = W \cdot H_{final}$ | Predicts one step at a time: $y_t = f(y_{t-1}, H_{context})$ |
| **Dependency** | **Independent:** Prediction at $t+2$ does not "know" about $t+1$. | **Dependent:** Prediction at $t+2$ is directly built upon $t+1$. |
| **Path Smoothness** | **Low:** Often jagged or noisy. | **High:** Naturally smooth and continuous. |
| **Context** | Uses only the *final* hidden state of the LSTM. | Uses **Attention** to access any part of the 5-minute history dynamically. |
| **Best For** | Single-point classification (e.g., Up/Down). | **Multi-step Time Series Forecasting (Path Generation).** |

**Summary:**
*   **DeepLOB** is like throwing a dart at a map. You pick a destination, but you don't draw the route.
*   **MHF** is like driving a car. You constantly adjust your steering based on where you are right now ($y_{t-1}$) and what you see in the mirror ($Attention$).
