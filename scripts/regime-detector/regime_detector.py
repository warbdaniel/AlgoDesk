import os
import yaml
import logging
import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Configuration & Logging Setup ---
LOG_DIR = '/opt/trading-desk/logs/system'
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'regime_detector.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Pydantic Data Models ---
class WebhookPayload(BaseModel):
    symbol: str
    timeframe: str = "15m"

# --- cTrader Data Fetcher ---
class BrokerClient:
    def __init__(self):
        self.client_id = os.getenv("CTRADER_CLIENT_ID", "YOUR_CLIENT_ID")
        self.client_secret = os.getenv("CTRADER_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
        self.access_token = os.getenv("CTRADER_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
        self.account_id = os.getenv("CTRADER_ACCOUNT_ID", "YOUR_ACCOUNT_ID")
        self._bot = None

    def _get_bot(self):
        if self._bot is None:
            try:
                from ctrader_sdk import CTraderBot
                self._bot = CTraderBot(
                    self.client_id,
                    self.client_secret,
                    self.access_token,
                    self.account_id
                )
            except Exception as e:
                logger.error(f"Failed to initialize CTraderBot: {e}")
                raise ValueError(f"cTrader SDK initialization failed: {e}")
        return self._bot

    def _map_timeframe(self, tf: str) -> str:
        mapping = {"1m": "m1", "5m": "m5", "15m": "m15", "30m": "m30", "1h": "h1", "4h": "h4", "1d": "d1"}
        return mapping.get(tf.lower(), "m15")

    def fetch_historical_bars(self, symbol: str, timeframe: str, bars_needed: int = 150) -> pd.DataFrame:
        tf_mapped = self._map_timeframe(timeframe)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=3)

        try:
            logger.info(f"Fetching {bars_needed} bars for {symbol} ({tf_mapped}) from cTrader...")
            bot = self._get_bot()
            df = bot.fetch_dataframe(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                timeframe=tf_mapped
            )
            df.columns = [c.lower() for c in df.columns]
            return df.tail(bars_needed)
        except Exception as e:
            logger.error(f"cTrader API Error: {str(e)}")
            raise ValueError(f"Failed to fetch data from broker: {str(e)}")

# --- Technical Indicators Math (Layer A) ---
def wilder_smoothing(series: pd.Series, periods: int) -> pd.Series:
    res = np.zeros_like(series.values)
    res[periods-1] = series[:periods].sum()
    for i in range(periods, len(series)):
        res[i] = res[i-1] - (res[i-1] / periods) + series.values[i]
    return pd.Series(res, index=series.index)

class ClassicalIndicators:
    @staticmethod
    def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close'].shift(1)
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        up_move, down_move = high - high.shift(1), low.shift(1) - low

        pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
        neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)

        tr_smooth = wilder_smoothing(tr, period)
        pos_di = 100 * (wilder_smoothing(pos_dm, period) / tr_smooth)
        neg_di = 100 * (wilder_smoothing(neg_dm, period) / tr_smooth)

        dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di))
        return dx.rolling(window=period).mean()

    @staticmethod
    def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close'].shift(1)
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calc_wma(series: pd.Series, period: int = 50) -> pd.Series:
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def calc_bb_width_percentile(close: pd.Series, period: int = 20, std_dev: int = 2, lookback: int = 100) -> pd.Series:
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        bbw = ((sma + (std * std_dev)) - (sma - (std * std_dev))) / sma
        return bbw.rolling(window=lookback).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)

# --- Regime Detector Engine ---
class RegimeDetector:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> dict:
        default_config = {
            "adx_trend_threshold": 25, "adx_range_threshold": 20,
            "bb_squeeze_percentile": 0.20, "wma_slope_lookback": 3
        }
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return default_config

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        adx = ClassicalIndicators.calc_adx(df, 14)
        atr = ClassicalIndicators.calc_atr(df, 14)
        atr_sma = atr.rolling(window=50).mean()
        wma_50 = ClassicalIndicators.calc_wma(df['close'], 50)
        bbw_pct = ClassicalIndicators.calc_bb_width_percentile(df['close'], 20, 2, 100)

        curr = {
            "adx": float(adx.iloc[-1]),
            "atr": float(atr.iloc[-1]),
            "atr_sma": float(atr_sma.iloc[-1]),
            "wma": float(wma_50.iloc[-1]),
            "wma_prev": float(wma_50.iloc[-(1 + self.config['wma_slope_lookback'])]),
            "bbw_pct": float(bbw_pct.iloc[-1])
        }

        direction = "BULL" if curr["wma"] > curr["wma_prev"] else "BEAR"
        volatility = "EXPANDING" if curr["atr"] > curr["atr_sma"] else "CONTRACTING"
        compression = curr["bbw_pct"] < self.config['bb_squeeze_percentile']

        confidence = 0.5
        if curr["adx"] > self.config["adx_trend_threshold"]:
            regime, confidence = ("STRONG_TREND", min(1.0, 0.6 + (curr["adx"] - 25) / 100)) if volatility == "EXPANDING" else ("MILD_TREND", 0.7)
        elif curr["adx"] < self.config["adx_range_threshold"]:
            regime, confidence = ("CHOPPY", 0.8) if compression else ("RANGING", 0.75)
        else:
            regime = "MILD_TREND" if volatility == "EXPANDING" else "RANGING"

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "direction": direction,
            "volatility": volatility,
            "indicators": {
                "adx": round(curr["adx"], 2),
                "atr_ratio": round(curr["atr"] / curr["atr_sma"], 2),
                "bb_percentile": round(curr["bbw_pct"], 2),
                "wma_normalized_slope": round((curr["wma"] - curr["wma_prev"]) / curr["wma_prev"] * 100, 4)
            }
        }

# --- FastAPI Setup ---
app = FastAPI(title="Regime Detector API", version="2.0.0")

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'config.yaml'
)
detector = RegimeDetector(config_path=config_path)
broker = BrokerClient()

@app.get('/health')
async def health_check():
    ctrader_configured = broker.client_id != "YOUR_CLIENT_ID"
    return {
        "status": "healthy",
        "service": "regime_detector",
        "version": "2.0.0",
        "phase": "Layer A",
        "framework": "FastAPI",
        "data_source": "cTrader" if ctrader_configured else "cTrader (NOT CONFIGURED)",
        "ctrader_ready": ctrader_configured
    }

@app.post('/regime')
async def get_regime(payload: WebhookPayload):
    try:
        df = await asyncio.to_thread(
            broker.fetch_historical_bars,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            bars_needed=150
        )

        if len(df) < 150:
            raise HTTPException(status_code=400, detail="cTrader returned insufficient historical data.")

        result = detector.detect(df)
        logger.info(f"[{payload.symbol}] Regime: {result['regime']} ({result['confidence']})")
        return result

    except ValueError as ve:
        logger.error(str(ve))
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during regime calculation.")

if __name__ == '__main__':
    logger.info("Starting FastAPI Regime Detector v2.0.0 on port 5000...")
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level="info")
