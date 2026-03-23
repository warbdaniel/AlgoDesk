
import os, json, yaml, logging, math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from flask import Flask, request, jsonify

LOG_DIR = '/opt/trading-desk/logs/system'
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, 'regime_detector.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def load_config(path='config.yaml'):
    dc = {'adx_trend_threshold': 25, 'adx_range_threshold': 20, 'bb_squeeze_percentile': 0.20, 'wma_slope_lookback': 3, 'atr_expansion_threshold': 1.0, 'min_periods': 150}
    try:
        with open(path) as f:
            uc = yaml.safe_load(f)
            if uc: dc.update(uc)
            logger.info(f'Config loaded from {path}')
    except FileNotFoundError:
        logger.warning(f'Config not found at {path}, using defaults.')
    return dc

def wilder_smooth(series, period):
    result = series.copy()
    fv = series.first_valid_index()
    if fv is None: return result
    loc = series.index.get_loc(fv)
    vs = series.iloc[loc:loc+period]
    if vs.isna().any() or len(vs) < period: return result
    si = loc + period - 1
    result.iloc[:si] = np.nan
    result.iloc[si] = vs.mean()
    for i in range(si+1, len(series)):
        v = series.iloc[i]
        if np.isnan(v): result.iloc[i] = result.iloc[i-1]
        else: result.iloc[i] = (result.iloc[i-1]*(period-1)+v)/period
    return result

def calc_adx(df, period=14):
    high, low, pc = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    um, dm = high-high.shift(1), low.shift(1)-low
    pdm = pd.Series(np.where((um>dm)&(um>0), um, 0.0), index=df.index)
    mdm = pd.Series(np.where((dm>um)&(dm>0), dm, 0.0), index=df.index)
    trs = wilder_smooth(tr, period)
    pds = wilder_smooth(pdm, period)
    mds = wilder_smooth(mdm, period)
    pdi = 100.0*(pds/trs)
    mdi = 100.0*(mds/trs)
    dis = pdi+mdi
    dis = dis.replace(0, np.nan)
    dx = 100.0*((pdi-mdi).abs()/dis)
    adx = wilder_smooth(dx, period)
    return adx, pdi, mdi

def calc_atr(df, period=14):
    high, low, pc = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    return wilder_smooth(tr, period)

def calc_wma(series, period=50):
    w = np.arange(1, period+1, dtype=float)
    return series.rolling(window=period).apply(lambda x: np.dot(x,w)/w.sum(), raw=True)

def calc_bb_width_percentile(close, bb_period=20, std_dev=2.0, lookback=100):
    sma = close.rolling(window=bb_period).mean()
    std = close.rolling(window=bb_period).std()
    bbw = ((sma+std*std_dev)-(sma-std*std_dev))/sma
    return bbw.rolling(window=lookback).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

class RegimeDetector:
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)

    def detect(self, df):
        mp = self.config['min_periods']
        if len(df) < mp: raise ValueError(f'Need {mp} periods, got {len(df)}')
        adx, pdi, mdi = calc_adx(df, 14)
        atr = calc_atr(df, 14)
        atr_sma = atr.rolling(window=50).mean()
        wma50 = calc_wma(df['close'], 50)
        bbp = calc_bb_width_percentile(df['close'], 20, 2.0, 100)
        c = self._safe_latest(adx, atr, atr_sma, wma50, bbp, pdi, mdi)
        if c is None: raise ValueError('Indicators produced NaN.')
        slb = self.config['wma_slope_lookback']
        wp = float(wma50.iloc[-(1+slb)])
        if math.isnan(wp) or wp == 0: raise ValueError('WMA lookback invalid.')
        ns = (c['wma']-wp)/wp*100
        ar = c['atr']/c['atr_sma'] if c['atr_sma'] != 0 else 1.0
        d = 'BULL' if ns > 0 else 'BEAR'
        v = 'EXPANDING' if ar > self.config['atr_expansion_threshold'] else 'CONTRACTING'
        comp = c['bbw_pct'] < self.config['bb_squeeze_percentile']
        regime, conf = self._classify(c, d, v, comp, ar, ns)
        result = {'regime': regime, 'confidence': round(conf, 2), 'direction': d, 'volatility': v,
            'indicators': {'adx': round(c['adx'],2), 'plus_di': round(c['plus_di'],2),
                'minus_di': round(c['minus_di'],2), 'atr': round(c['atr'],6),
                'atr_ratio': round(ar,2), 'bb_percentile': round(c['bbw_pct'],2),
                'wma_normalized_slope': round(ns,4), 'wma_50': round(c['wma'],6)}}
        logger.info(f"Regime: {regime} | Conf: {conf:.2f} | Dir: {d} | Vol: {v} | ADX: {c['adx']:.1f}")
        return result

    def _safe_latest(self, adx, atr, atr_sma, wma50, bbp, pdi, mdi):
        vals = {'adx': float(adx.iloc[-1]), 'atr': float(atr.iloc[-1]), 'atr_sma': float(atr_sma.iloc[-1]),
            'wma': float(wma50.iloc[-1]), 'bbw_pct': float(bbp.iloc[-1]),
            'plus_di': float(pdi.iloc[-1]), 'minus_di': float(mdi.iloc[-1])}
        for k,vl in vals.items():
            if math.isnan(vl):
                logger.error(f'NaN in {k}')
                return None
        return vals

    def _classify(self, c, d, v, comp, ar, ns):
        adx = c['adx']
        tt = self.config['adx_trend_threshold']
        rt = self.config['adx_range_threshold']
        if adx > tt:
            s = 0.4 + min(0.2, (adx-tt)/125)
            if v == 'EXPANDING': s += 0.2
            if abs(c['plus_di']-c['minus_di']) > 10: s += 0.1
            if abs(ns) > 0.05: s += 0.1
            if v == 'EXPANDING' and s >= 0.7: return 'STRONG_TREND', min(1.0, s)
            else: return 'MILD_TREND', min(1.0, s*0.9)
        elif adx < rt:
            s = 0.4 + min(0.15, (rt-adx)/100)
            if v == 'CONTRACTING': s += 0.15
            if abs(c['plus_di']-c['minus_di']) < 5: s += 0.15
            if abs(ns) < 0.03: s += 0.15
            if comp: return 'CHOPPY', min(1.0, s+0.1)
            else: return 'RANGING', min(1.0, s)
        else:
            if v == 'EXPANDING' and abs(ns) > 0.05: return 'MILD_TREND', 0.45
            elif comp: return 'CHOPPY', 0.45
            else: return 'RANGING', 0.40

app = Flask(__name__)
detector = RegimeDetector()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status':'healthy','service':'regime_detector','version':'1.1.0','phase':'Layer A'}), 200

@app.route('/regime', methods=['POST'])
def get_regime():
    try:
        p = request.get_json()
        if not p or 'data' not in p:
            return jsonify({'error':'Expected {data: [ohlcv]}'}), 400
        metadata = p.get('metadata', {})
        df = pd.DataFrame(p['data'])
        df.columns = [c.lower() for c in df.columns]
        req = {'open','high','low','close','volume'}
        if not req.issubset(set(df.columns)):
            return jsonify({'error':f'Missing cols. Need: {req}'}), 400
        for col in req:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        result = detector.detect(df)
        if metadata:
            result['metadata'] = metadata
            logger.info(f"Symbol: {metadata.get('symbol','?')} | TF: {metadata.get('timeframe','?')}")
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error':str(e)}), 400
    except Exception as e:
        logger.error(f'API Error: {str(e)}', exc_info=True)
        return jsonify({'error':f'Internal: {str(e)}'}), 500

def run_synthetic_test():
    print(chr(10)+'='*60)
    print('  REGIME DETECTOR - SYNTHETIC DATA TEST')
    print('='*60)
    np.random.seed(42)
    n = 250
    ret = np.random.normal(0.001, 0.01, n)
    cl = 100*np.exp(np.cumsum(ret))
    hi = cl*(1+np.random.uniform(0.001,0.015,n))
    lo = cl*(1-np.random.uniform(0.001,0.015,n))
    op = np.roll(cl,1); op[0]=100.0
    vo = np.random.randint(1000,10000,n)
    df = pd.DataFrame({'open':op,'high':hi,'low':lo,'close':cl,'volume':vo})
    td = RegimeDetector()
    try:
        r = td.detect(df)
        print(chr(10)+'Test PASSED. Output:')
        print(json.dumps(r, indent=2))
        print('='*60)
        return True
    except Exception as e:
        print(f'Test FAILED: {e}')
        print('='*60)
        return False

if __name__ == '__main__':
    ok = run_synthetic_test()
    if ok:
        logger.info('Starting Regime Detector API on port 5000...')
        print(chr(10)+'Starting Flask API on http://0.0.0.0:5000')
        print('Endpoints: GET /health | POST /regime'+chr(10))
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print('Fix errors before starting API.')
