# Project Summary: Can You Achieve the Best Price?

Implementing effective trading strategies, especially at higher frequencies, is a complex task that delves into the intricate world of market microstructure. Even when trading relatively small volumes, the actual execution price is influenced by a multitude of dynamic factors like spread, slippage, and adverse selection. In this complex setting, the ability to achieve a favorable price is a rare skill that can significantly impact a strategy’s overall performance.

At Ultramarin, we develop quantitative trading strategies for global equity and futures markets. Consequently, possessing a profound understanding of the actual costs associated with trade implementation is of paramount importance to the success and viability of any strategy we deploy. Intriguingly, parallels can be drawn between the market microstructure of futures and that of cryptocurrencies, a similarity that provides valuable insights for developing robust execution strategies across diverse asset classes.

To help us better understand the complex structure behind strategy implementation, we have compiled a detailed, high-frequency dataset focusing on the most liquid cryptocurrency pairs. We are particularly interested in studying implementation algorithms that operate under the assumption of trading a given strategy at the close of every hour. Therefore, your challenge is to develop an algorithm capable of filling a specified trade volume within the last minute of every hour, striving to achieve the best possible execution price.

## 1. Challenge Details

### 1.1. The Data

For each cryptocurrency pair, you will be provided with three distinct datasets, each offering data at second-level granularity:

- X_train.parquet: This dataset contains Level 2 (L2) order book data, providing a snapshot of the first 5 levels of the order book per second, see Appendix 4 below for a short introduction to the order book and [1], [2]. It is augmented with standard candlestick data (high, low, open, close, and volume) for each corresponding second. X_train encompasses data from the beginning of the hour (00:00 in mm:ss format) up to the 58:59 second.
- Y_train.parquet: This dataset shares the same structural schema as X_train.parquet but specifically contains data for the final minute of each hour, spanning from 59:00 to 59:59. This segment represents the critical period for your execution strategy.
- X_test.parquet: This dataset mirrors the structure and content of X_train.parquet and is provided for testing your algorithm’s performance on unseen data.

In addition to these datasets, you will be supplied with three supplementary files crucial for the challenge:

- Vol_to_fill.txt: This file specifies the precise amount of base currency volume that needs to be filled at the end of each minute. Your algorithm must ensure this target volume is met.
- Example.ipynb: A Jupyter Notebook containing illustrative code examples and further detailed explanations. It is highly recommended to thoroughly review this notebook before attempting the challenge.
- simulate_walk_the_book.py: A Python file containing the function to simulate the execution process, with tests.

### 1.2. The Task

Your practical task involves generating 60 distinct execution positions for each hour. These positions are designed to cumulatively fill the specified order volume within the last minute of every hour. We further request that you generate 60 positions for the ask side (representing buying the asset) and 60 positions for the buy side (representing selling the asset). This duality ensures a comprehensive evaluation of your strategy across both market directions.

### 1.3. Anonymization

The data have been anonymized: the timestamps for each hour have been converted into a numerical string, referred to as an “anonymized ID”. Rows sharing the same “anonymized ID” correspond to the same universal timestamp across different currency pairs. Additionally, the actual price values have been transformed to discourage external information leakage!

## 2. Evaluation Metric

Numerous methodologies exist for quantifying the performance of an implementation algorithm. In the context of this challenge, our primary focus is to assess the normalized difference between your average execution price and the closing price of the hour. More formally, your objective is to minimize, on average, the following quantity:

$$
\text{Minimization Objective: } \frac{|\text{average\_price} - \text{close\_price}|}{\text{close\_price}}
$$

where:

- close_price: This denotes the final traded price of the hour (the data is chosen so that this is always well defined). It is extracted, in polars, from the provided data as:

  close_price := y_last_min.select("close").drop_nulls().to.series().last()

  This effectively represents the market’s closing price for that specific hourly period.

- average_price: This represents the Volume-Weighted Average Price (VWAP) of your executed trades for the hour. It is computed as the total value of all executed trades divided by their aggregate volume:

  average_price := total_value_executed / total_volume_executed

The total_value_executed term is a running sum, accumulating the value of each sub-execution:

$$
\text{total\_value\_executed} := \text{total\_value\_executed} + (\text{position\_to\_fill} \times \text{price})
$$

For instance: If you execute a trade for 0.5 units of volume at a price of 10 and subsequently another trade for 0.5 units of volume at a price of 20, the resulting average_price (VWAP) for these combined executions would be 15. This effectively averages the prices weighted by the volume traded at each price.

## 3. Caveats and Assumptions

It is crucial to acknowledge that this challenge operates under a simplified model of real-world trading strategy implementation. In practice, nobody can definitively predict how the market would have reacted had a different order been executed or at a different time. Therefore, we operate under the fundamental assumption of a “replenishing order book”.

This assumption implies that any actions taken at second s (i.e., your executed trades) do not instantaneously affect the state of the order book at the subsequent second s + 1. While this might be (and often is) a somewhat unrealistic assumption for true high-frequency trading scenarios, it serves as a reasonable approximation for slower implementations of strategies on highly liquid assets within traditional equity markets.

### 3.1. Handling Unfilled Positions

Furthermore, since the y_test dataset (which represents the real market outcomes) is hidden, it is possible that a position at second s might not be completely filled, even if the entirety of the available order book at that second is “walked” (i.e., fully consumed) or due to data unavailability within the dataset. In such events, any unfilled portion of the position will be automatically carried forward and added to the volume required at second s + 1. In order to assist you in evaluating your models, a function to implement the given positions is provided in the .py file.

### 3.2. Extra Penalty: Unfilled Volume

The actual scoring function also incorporates an additional penalty. This penalty is directly proportional to the ratio of the remaining volume_to_fill at the end of the period to the total_volume_executed. This mechanism encourages algorithms that not only achieve a good average price but also effectively fulfill the entire target volume.

$$
\text{Volume Penalty} := \min \left( 100, \frac{\text{volume\_to\_fill}}{\text{total\_volume\_executed}} \right)
$$

## 4. Bonus Challenge

As an additional challenge, for the ask side, can you modify your approach to minimize the raw difference:

$$
\text{Bonus Objective: } \frac{\text{average\_price} - \text{close\_price}}{\text{close\_price}} \tag{1}
$$

If your implementation consistently achieves a negative value for this metric, it indicates that your strategy is executing at an average price lower than the closing price, which is characteristic of a highly effective high-frequency trading strategy! Of course, you can maximize (1) for the bid side.

## Appendix: Order Book Crash Course

At its core, an order book is a real-time electronic list of all outstanding buy and sell orders for a particular financial instrument (such as a stock or cryptocurrency) on an exchange. It can be conceptualized as a dynamic waiting list for market participants:

- Bids: These represent orders placed by buyers, specifying the maximum price they are willing to pay for a designated quantity of the asset. The highest bid price is always positioned at the top of the bid side of the order book.
- Asks (or Offers): These represent orders placed by sellers, specifying the minimum price they are willing to accept for a designated quantity of the asset. The lowest ask price is always positioned at the top of the ask side of the order book.
- The variables ask_price_i and ask_vol_i refer to the price and quantity of the i-th (or i-th lowest) ask price, respectively.
- Similarly, the variables bid_price_i and bid_vol_i refer to the price and quantity of the i-th (or i-th highest) bid price, respectively.

This format allows for direct access to specific order book levels within the dataset.

### How the Order Book Works

When a buy order is submitted, it first attempts to execute against the lowest available ask price (the best offer). If the buy order’s volume exceeds the available volume at that price level, it consumes that volume and then proceeds to match against the next lowest ask price, and so forth, until the entire order is filled or all available liquidity at sequentially higher prices is exhausted (also known as “walking the book”).

Conversely, a sell order attempts to execute against the highest available bid price (the best bid), moving down through deeper bid levels as necessary. If an order cannot be immediately filled at its specified price, it is typically added to the order book as a new resting bid or ask order at its declared price level.

### Understanding Level 2 (L2) Data

Level 2 (L2) order book data provides a significantly more granular and comprehensive view into market depth compared to Level 1 data. While Level 1 data is limited to showing only the single best bid and best ask prices (often referred to as the “top of the book”), L2 data reveals multiple price levels on both the bid and ask sides. For each of these visible levels, the total aggregated volume available at that specific price is also displayed.

For example, L2 data might show not only the best bid at $100 with 50 units of volume, but immediately beneath it, the next best bid at $99.90 with 100 units, followed by $99.85 with 75 units, and so on, extending several levels deep into the order book. This comprehensive information about market “depth” is invaluable for traders, offering insights into immediate market liquidity, the distribution of supply and demand, and potential imbalances that could foreshadow future price movements.

