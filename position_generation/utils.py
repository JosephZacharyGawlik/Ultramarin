import numpy as np
import pandas as pd
from Ultramarin.data.simulate_walk_the_book import simulate_walk_the_book
import warnings

class PositionPredictor:

    def __init__(self, y_train, ask_price_cols, ask_vol_cols, bid_price_cols, bid_vol_cols, volume_to_fill, side="bid"):

        """
        Bid side means we're selling (consuming bids). Ask side means we're buying.
        """

        # store data and config
        self.y_train = y_train.copy()
        self.train_anon_ids = sorted(self.y_train["anonymized_id"].unique())
        self.train_times = pd.DataFrame(sorted(self.y_train["time_in_hour"].unique()), columns=["time_in_hour"])
        self.ask_price_cols = ask_price_cols
        self.ask_vol_cols = ask_vol_cols
        self.bid_price_cols = bid_price_cols
        self.bid_vol_cols = bid_vol_cols
        self.volume_to_fill = volume_to_fill
        self.side = side

        # add relevant metrics to y_train

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # mid price proportion weights
        self.y_train["mid"] = (self.y_train["ask_price_1"] + self.y_train["bid_price_1"]) / 2

        mid_mean = self.y_train.groupby("anonymized_id")["mid"].transform("mean")
        mid_final = self.y_train.groupby("anonymized_id")["mid"].transform("last")

        # raw distance-to-close weights
        raw_w = 1 - np.abs(self.y_train["mid"] - mid_final) / mid_mean

        # normalize to [0,1] per instrument
        w_min = raw_w.groupby(self.y_train["anonymized_id"]).transform("min")
        w_max = raw_w.groupby(self.y_train["anonymized_id"]).transform("max")

        norm_w = (raw_w - w_min) / (w_max - w_min)

        self.y_train["mid_prop"] = norm_w / norm_w.groupby(self.y_train["anonymized_id"]).transform("sum")

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # close price proportion weights
        close_mean = self.y_train.groupby("anonymized_id")["close"].transform("mean")
        close_final = self.y_train.groupby("anonymized_id")["close"].transform("last")

        # raw distance-to-close weights
        raw_w = 1 - np.abs(self.y_train["close"] - close_final) / close_mean

        # normalize to [0,1] per instrument
        w_min = raw_w.groupby(self.y_train["anonymized_id"]).transform("min")
        w_max = raw_w.groupby(self.y_train["anonymized_id"]).transform("max")

        norm_w = (raw_w - w_min) / (w_max - w_min)

        self.y_train["close_prop"] = norm_w / norm_w.groupby(self.y_train["anonymized_id"]).transform("sum")

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # micro price proportion weights
        self.y_train["micro_price"] = (
            self.y_train["ask_price_1"] * self.y_train["bid_vol_1"] +
            self.y_train["bid_price_1"] * self.y_train["ask_vol_1"]
        ) / (self.y_train["ask_vol_1"] + self.y_train["bid_vol_1"])


        micro_mean = self.y_train.groupby("anonymized_id")["micro_price"].transform("mean")
        micro_final = self.y_train.groupby("anonymized_id")["micro_price"].transform("last")

        raw_w = 1 - np.abs(self.y_train["micro_price"] - micro_final) / micro_mean

        w_min = raw_w.groupby(self.y_train["anonymized_id"]).transform("min")
        w_max = raw_w.groupby(self.y_train["anonymized_id"]).transform("max")

        norm_w = (raw_w - w_min) / (w_max - w_min)

        self.y_train["micro_prop"] = (
            norm_w /
            norm_w.groupby(self.y_train["anonymized_id"]).transform("sum")
        )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # volume proportion weights (VWAP)
        self.y_train["vwap_prop"] = self.y_train["volume"] / self.y_train.groupby("anonymized_id")["volume"].transform("sum")

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # order flow imbalance (OFI) proportion weights
        self.y_train["ofi"] = self.y_train.groupby("anonymized_id")["bid_vol_1"].diff() - self.y_train.groupby("anonymized_id")["ask_vol_1"].diff()
        self.y_train["ofi"] = self.y_train["ofi"].fillna(0)

        if self.side == "ask":
            self.y_train["ofi"] = -self.y_train["ofi"].clip(upper=0)
        elif self.side == "bid":
            self.y_train["ofi"] = self.y_train["ofi"].clip(lower=0) # we only care about positive OFI for bids. positive OFI means more bid volume means more bids to consume
        else:
            raise ValueError("side must be 'bid' or 'ask'")
        
        raw_w = self.y_train["ofi"]

        w_min = raw_w.groupby(self.y_train["anonymized_id"]).transform("min")
        w_max = raw_w.groupby(self.y_train["anonymized_id"]).transform("max")

        norm_w = (raw_w - w_min) / (w_max - w_min)

        self.y_train["ofi_prop"] = (
            norm_w /
            norm_w.groupby(self.y_train["anonymized_id"]).transform("sum")
        )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # order book imbalance (OBI) proportion weights
        self.y_train["obi"] = (self.y_train["bid_vol_1"] - self.y_train["ask_vol_1"]) / (self.y_train["bid_vol_1"] + self.y_train["ask_vol_1"])

        if self.side == "ask":
            self.y_train["obi"] = -self.y_train["obi"].clip(upper=0)
        elif self.side == "bid":
            self.y_train["obi"] = self.y_train["obi"].clip(lower=0) 
        else:
            raise ValueError("side must be 'bid' or 'ask'")
        
        raw_w = self.y_train["obi"]

        w_min = raw_w.groupby(self.y_train["anonymized_id"]).transform("min")
        w_max = raw_w.groupby(self.y_train["anonymized_id"]).transform("max")

        norm_w = (raw_w - w_min) / (w_max - w_min)

        self.y_train["obi_prop"] = (
            norm_w /
            norm_w.groupby(self.y_train["anonymized_id"]).transform("sum")
        )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # spread weights
        self.y_train["spread"] = self.y_train["ask_price_1"] - self.y_train["bid_price_1"]

        raw_w = 1 / self.y_train["spread"]

        w_min = raw_w.groupby(self.y_train["anonymized_id"]).transform("min")
        w_max = raw_w.groupby(self.y_train["anonymized_id"]).transform("max")

        norm_w = (raw_w - w_min) / (w_max - w_min)

        self.y_train["spread_prop"] = (
            norm_w /
            norm_w.groupby(self.y_train["anonymized_id"]).transform("sum")
        )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # build per-instrument dataframes
        self.df_insts = {anon_id: df_inst.sort_values("time_in_hour") for anon_id, df_inst in self.y_train.groupby("anonymized_id")}


    def compute_positions(self, strategy, weighting="uniform", K_seconds=60):
            
            assert weighting in ["uniform", "linear", "quadratic"]

            positions = np.zeros((len(self.train_anon_ids), 60))

            if type(strategy) == list:

                assert np.isclose(sum([s["fraction"] for s in strategy]), 1)

                for strat in strategy:
                    positions += strat["fraction"] * self.compute_positions(strat["strategy"], weighting=strat["weighting"], K_seconds=strat["K_seconds"])
                return positions

            for i, anon_id in enumerate(self.train_anon_ids):

                df_inst = self.df_insts[anon_id]

                if strategy == "baseline":

                    weights = np.ones(K_seconds)

                elif strategy == "mid":

                    weights = df_inst["mid_prop"].values[-K_seconds:].copy()

                elif strategy == "vwap":

                    weights = df_inst["vwap_prop"].values[-K_seconds:].copy()

                elif strategy == "micro":

                    weights = df_inst["micro_prop"].values[-K_seconds:].copy()

                elif strategy == "ofi":

                    weights = df_inst["ofi_prop"].values[-K_seconds:].copy()

                elif strategy == "obi":

                    weights = df_inst["obi_prop"].values[-K_seconds:].copy()

                elif strategy == "spread":

                    weights = df_inst["spread_prop"].values[-K_seconds:].copy()

                elif strategy == "close":

                    weights = df_inst["close_prop"].values[-K_seconds:].copy()

                else:
                    raise ValueError("strategy must be 'baseline', 'mid', 'vwap', 'micro', 'ofi', 'obi', 'spread', or 'close'")
                
                if weighting == "linear":
                    weights *= np.linspace(1, K_seconds, K_seconds)
                elif weighting == "quadratic":
                    weights *= np.square(np.linspace(0, K_seconds, K_seconds))

                if weights.sum() == 0 or not np.isfinite(weights).all():
                            weights = np.ones(K_seconds) / K_seconds
                else:
                    weights = weights / weights.sum()

                positions[i, -K_seconds:] = self.volume_to_fill * weights

            if self.side == "bid":
                positions = -positions

            return positions


    def compute_execution_metrics(self, strategy, weighting="uniform", K_seconds=60, positions=None):

        """
        Compute implementation error metrics for baseline, mid, or vwap strategy.

        Parameters
        ----------
        strategy : str
            One of ["baseline", "mid", "vwap"]
        K_seconds : int
            Number of seconds at end of minute to execute trades
        weighting : str
            One of ["uniform", "linear", "quadratic"]
        """

        if K_seconds <= 0 or K_seconds > 60:
            raise ValueError("K_seconds must be between 1 and 60")

        # compute positions for each strategy
        if positions is None:
            positions = self.compute_positions(strategy, weighting, K_seconds)

        # --- compute implementation error ---
        bps = []

        for i, anon_id in enumerate(self.train_anon_ids):

            df_inst = self.df_insts[anon_id]

            ask_prices = df_inst[self.ask_price_cols].to_numpy()
            ask_vols = df_inst[self.ask_vol_cols].to_numpy()
            bid_prices = df_inst[self.bid_price_cols].to_numpy()
            bid_vols = df_inst[self.bid_vol_cols].to_numpy()

            close_price = df_inst["close"].iloc[-1]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                vol, avg_price = simulate_walk_the_book(
                    positions[i],
                    ask_prices,
                    ask_vols,
                    bid_prices,
                    bid_vols,
                )

            if vol > 0 and not np.isnan(avg_price):

                impl_error = abs(avg_price - close_price) / close_price * 10000
                vol_penalty = min(100.0, self.volume_to_fill / vol)

                bps.append(impl_error * vol_penalty)

        bps = np.array(bps)

        metrics = {
            "strategy": strategy,
            "K_seconds": K_seconds,
            "instruments": len(bps),
            "mean_bps": bps.mean(),
            "median_bps": np.median(bps),
            "std_bps": bps.std(),
            "min_bps": bps.min(),
            "max_bps": bps.max(),
            "bps": bps,
            "positions": positions
        }

        return metrics
