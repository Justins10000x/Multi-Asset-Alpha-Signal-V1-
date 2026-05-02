#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  SYSTEMATIC EDGE SCANNER
  Scans 180+ signal combinations across SPY, QQQ, IWM
  Tests mean reversion, calendar anomalies, cross-asset,
  RSI extremes, MA distance, volatility regimes
═══════════════════════════════════════════════════════════════

  USAGE:
    python edge_scanner.py --spy SPY.csv --vix VIX.csv --qqq QQQ.csv --iwm IWM.csv --tnx TNX.csv
    python edge_scanner.py --spy SPY.csv --vix VIX.csv   # minimal run
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from scipy import stats
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
OOS_CUTOFF      = '2024-01-01'   # in/out-of-sample split date
MIN_TRADES      = 15             # minimum trades to report a signal
TP_DEFAULT      = 0.005          # default take profit (0.50%)
SL_DEFAULT      = 0.003          # default stop loss  (0.30%)
OOS_SHARPE_MIN  = 1.5            # minimum OOS Sharpe to be "validated"
OOS_WR_MIN      = 52.0           # minimum OOS win rate to be "validated"


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_investing_csv(path, name=''):
    """Load CSV exported from Investing.com format."""
    df = pd.read_csv(path, thousands=',')
    df.columns = df.columns.str.strip().str.replace('"', '')
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.replace('"', ''), format='mixed')
    for col in ['Price', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Vol.' in df.columns:
        def parse_vol(v):
            v = str(v).replace('"', '').replace(',', '').strip()
            if 'M' in v: return float(v.replace('M', '')) * 1e6
            if 'K' in v: return float(v.replace('K', '')) * 1e3
            try: return float(v)
            except: return np.nan
        df['Volume'] = df['Vol.'].apply(parse_vol)
    df = df.rename(columns={'Price': 'Close'})
    keep = ['Date', 'Open', 'High', 'Low', 'Close'] + (['Volume'] if 'Volume' in df.columns else [])
    df = df[keep].dropna(subset=['Close']).sort_values('Date').reset_index(drop=True)
    if name:
        print(f"  {name}: {len(df)} rows | {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_spy_nasdaq_csv(path, name='SPY'):
    """Load CSV exported from Nasdaq/Yahoo format (Close/Last column)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if 'Close/Last' in df.columns:
        df.rename(columns={'Close/Last': 'Close'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"  {name}: {len(df)} rows | {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_vix_csv(path):
    """Load CBOE VIX history CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    date_col  = next((c for c in df.columns if 'DATE' in c.upper() or 'Date' in c), df.columns[0])
    close_col = next((c for c in df.columns if 'CLOSE' in c.upper() or 'Close' in c), df.columns[-1])
    df = df.rename(columns={date_col: 'Date', close_col: 'VIX'})
    df['Date'] = pd.to_datetime(df['Date'])
    df['VIX']  = pd.to_numeric(df['VIX'], errors='coerce')
    return df[['Date', 'VIX']].dropna()


# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df, vix_df, tnx_df=None):
    """Add all signal features to OHLCV dataframe."""
    d = df.copy()
    d['Return']     = d['Close'].pct_change()
    d['Return2d']   = d['Return'] + d['Return'].shift(1)
    d['Return3d']   = d['Return2d'] + d['Return'].shift(2)
    d['MA20']       = d['Close'].rolling(20).mean()
    d['MA50']       = d['Close'].rolling(50).mean()
    d['MA200']      = d['Close'].rolling(200).mean()
    d['Above200']   = (d['Close'] > d['MA200']).astype(int)
    d['Above50']    = (d['Close'] > d['MA50']).astype(int)
    d['Dist200']    = (d['Close'] - d['MA200']) / d['MA200']
    d['ATR14']      = (d['High'] - d['Low']).rolling(14).mean()
    d['ATR_pct']    = d['ATR14'] / d['Close']
    d['RSI']        = compute_rsi(d['Close'], 14)
    d['DayOfWeek']  = d['Date'].dt.dayofweek
    d['DayOfMonth'] = d['Date'].dt.day
    d['Month']      = d['Date'].dt.month
    d['TOM']        = (d['DayOfMonth'] <= 3).astype(int)
    d['EOM']        = (d['DayOfMonth'] >= 27).astype(int)

    if 'Volume' in d.columns:
        d['VolRatio'] = d['Volume'] / d['Volume'].rolling(20).mean()
    else:
        d['VolRatio'] = 1.0

    # Consecutive down/up day signals
    for thresh in [0.002, 0.003, 0.005, 0.008, 0.010]:
        t = int(thresh * 1000)
        d[f'Down{t}_1d'] = (d['Return'] <= -thresh).astype(int)
        d[f'Down{t}_2d'] = ((d['Return'] <= -thresh) & (d['Return'].shift(1) <= -thresh)).astype(int)
        d[f'Down{t}_3d'] = ((d['Return'] <= -thresh) & (d['Return'].shift(1) <= -thresh)
                            & (d['Return'].shift(2) <= -thresh)).astype(int)
        d[f'Up{t}_1d']   = (d['Return'] >= thresh).astype(int)
        d[f'Up{t}_2d']   = ((d['Return'] >= thresh) & (d['Return'].shift(1) >= thresh)).astype(int)

    # Merge VIX
    d = d.merge(vix_df[['Date', 'VIX']], on='Date', how='left')
    d['VIX'] = d['VIX'].ffill()
    d['VIX_Low']   = (d['VIX'] < 15).astype(int)
    d['VIX_Mid']   = ((d['VIX'] >= 15) & (d['VIX'] < 25)).astype(int)
    d['VIX_High']  = (d['VIX'] >= 25).astype(int)
    d['VIX_Spike'] = (d['VIX'] / d['VIX'].rolling(5).mean() > 1.3).astype(int)

    # Merge TNX (10-year yield) if provided
    if tnx_df is not None:
        tnx_m = tnx_df[['Date', 'Close']].rename(columns={'Close': 'TNX'}).copy()
        tnx_m['TNX_chg']   = tnx_m['TNX'].pct_change()
        tnx_m['TNX_2d']    = tnx_m['TNX_chg'] + tnx_m['TNX_chg'].shift(1)
        tnx_m['TNX_spike'] = (tnx_m['TNX_2d'] > 0.03).astype(int)
        tnx_m['TNX_drop']  = (tnx_m['TNX_2d'] < -0.03).astype(int)
        d = d.merge(tnx_m[['Date', 'TNX', 'TNX_chg', 'TNX_spike', 'TNX_drop']],
                    on='Date', how='left')

    # Targets
    d['Next_Return'] = d['Return'].shift(-1)
    d['Next_Open']   = d['Open'].shift(-1)
    d['Next_High']   = d['High'].shift(-1)
    d['Next_Low']    = d['Low'].shift(-1)
    d['Next_Close']  = d['Close'].shift(-1)

    return d.dropna()


# ─────────────────────────────────────────────────────────────
#  SIGNAL TESTING ENGINE
# ─────────────────────────────────────────────────────────────

def test_signal(df, signal_col, label, min_n=MIN_TRADES):
    """Test a binary signal column for predictive edge."""
    sub      = df[df[signal_col] == 1]
    baseline = df['Next_Return']
    if len(sub) < min_n:
        return None
    sig_ret = sub['Next_Return']
    _, p    = stats.ttest_1samp(sig_ret, 0)
    sharpe  = (sig_ret.mean() / sig_ret.std() * np.sqrt(252)
               if sig_ret.std() > 0 else 0)
    return {
        'Signal':    label,
        'N':         len(sub),
        'WinRate':   (sig_ret > 0).mean() * 100,
        'AvgReturn': sig_ret.mean() * 100,
        'Baseline':  baseline.mean() * 100,
        'Edge':      (sig_ret.mean() - baseline.mean()) * 100,
        'p_val':     p,
        'Sharpe':    sharpe,
    }


def run_backtest(df, signal_col, tp=TP_DEFAULT, sl=SL_DEFAULT):
    """Simulate trades with TP/SL/time-stop bracket execution."""
    cutoff = pd.Timestamp(OOS_CUTOFF)
    trades = []
    for i in range(len(df) - 1):
        if df.iloc[i][signal_col] != 1:
            continue
        nxt   = df.iloc[i + 1]
        entry = nxt['Open']
        tp_p  = entry * (1 + tp)
        sl_p  = entry * (1 - sl)
        hit_tp = nxt['High'] >= tp_p
        hit_sl = nxt['Low']  <= sl_p
        if hit_tp and hit_sl:
            pnl = tp if (nxt['High'] - entry) <= (entry - nxt['Low']) else -sl
        elif hit_tp:
            pnl = tp
        elif hit_sl:
            pnl = -sl
        else:
            pnl = (nxt['Close'] - entry) / entry
        trades.append({
            'Date':   nxt['Date'],
            'PnL':    pnl,
            'Win':    pnl > 0,
            'Sample': 'IN' if nxt['Date'] < cutoff else 'OUT',
        })
    return pd.DataFrame(trades) if trades else pd.DataFrame()


def summarise_backtest(t, label=''):
    """Print backtest statistics for IN, OUT and ALL samples."""
    if t.empty:
        print(f"  {label}: No trades.")
        return {}
    results = {}
    for sample in ['IN', 'OUT', 'ALL']:
        sub = t if sample == 'ALL' else t[t['Sample'] == sample]
        if len(sub) < 3:
            continue
        wr    = sub['Win'].mean() * 100
        avg   = sub['PnL'].mean() * 100
        total = sub['PnL'].sum() * 100
        sharpe = (sub['PnL'].mean() / sub['PnL'].std() * np.sqrt(252)
                  if sub['PnL'].std() > 0 else 0)
        cum = sub['PnL'].cumsum()
        mdd = (cum.cummax() - cum).max() * 100
        results[sample] = {'N': len(sub), 'WR': wr, 'Avg': avg,
                           'Total': total, 'Sharpe': sharpe, 'MDD': mdd}
    return results


# ─────────────────────────────────────────────────────────────
#  FULL SCANNER
# ─────────────────────────────────────────────────────────────

def run_full_scan(frames: dict) -> pd.DataFrame:
    """
    frames: dict of {ticker_name: feature_df}
    Returns ranked DataFrame of all signals tested.
    """
    results = []

    for ticker, df in frames.items():
        # Category 1 — Raw mean reversion signals
        for col in [c for c in df.columns if c.startswith(('Down', 'Up'))]:
            r = test_signal(df, col, f'{ticker} | {col}')
            if r:
                results.append({**r, 'Ticker': ticker, 'Category': 'MeanReversion'})

        # Category 2 — VIX-conditioned
        for vix_col in ['VIX_Low', 'VIX_Mid', 'VIX_High', 'VIX_Spike']:
            for down_col in ['Down3_2d', 'Down5_2d', 'Down3_1d', 'Down3_3d']:
                if down_col not in df.columns:
                    continue
                df['_tmp'] = ((df[down_col] == 1) & (df[vix_col] == 1)).astype(int)
                r = test_signal(df, '_tmp', f'{ticker} | {down_col}+{vix_col}')
                if r:
                    results.append({**r, 'Ticker': ticker, 'Category': 'VIX-Conditioned'})

        # Category 3 — Calendar anomalies
        calendar_signals = {
            'Turn-of-Month':    df['TOM'],
            'End-of-Month':     df['EOM'],
            'Monday':           (df['DayOfWeek'] == 0).astype(int),
            'Friday':           (df['DayOfWeek'] == 4).astype(int),
            'OpEx-Week':        ((df['DayOfMonth'] >= 15) & (df['DayOfMonth'] <= 21)).astype(int),
            'January':          (df['Month'] == 1).astype(int),
            'May-Sep':          df['Month'].isin([5, 6, 7, 8, 9]).astype(int),
        }
        for name, sig in calendar_signals.items():
            df['_tmp'] = sig
            r = test_signal(df, '_tmp', f'{ticker} | {name}')
            if r:
                results.append({**r, 'Ticker': ticker, 'Category': 'Calendar'})

        # Category 4 — RSI extremes
        for rsi_lo in [25, 30, 35, 40]:
            df['_tmp'] = (df['RSI'] < rsi_lo).astype(int)
            r = test_signal(df, '_tmp', f'{ticker} | RSI<{rsi_lo}')
            if r:
                results.append({**r, 'Ticker': ticker, 'Category': 'RSI'})
        for rsi_hi in [60, 65, 70, 75]:
            df['_tmp'] = (df['RSI'] > rsi_hi).astype(int)
            r = test_signal(df, '_tmp', f'{ticker} | RSI>{rsi_hi}')
            if r:
                results.append({**r, 'Ticker': ticker, 'Category': 'RSI'})

        # Category 5 — MA distance extremes
        for pct in [-0.03, -0.05, -0.08]:
            df['_tmp'] = (df['Dist200'] < pct).astype(int)
            r = test_signal(df, '_tmp', f'{ticker} | >{abs(pct)*100:.0f}%_below_200MA')
            if r:
                results.append({**r, 'Ticker': ticker, 'Category': 'MA-Distance'})

        # Category 6 — Composite / validated signals
        composite = {
            'SignalB_style':     ((df['Down3_2d'] == 1) & (df['VolRatio'] < 1.0) & (df['VIX'] < 25)),
            'RSI35_plus_Down':   ((df['RSI'] < 35) & (df['Down3_1d'] == 1)),
            'TOM_Above200':      ((df['TOM'] == 1) & (df['Above200'] == 1)),
            'Down2d_A200_LowVIX':((df['Down3_2d'] == 1) & (df['Above200'] == 1) & (df['VIX_Low'] == 1)),
        }
        for name, sig in composite.items():
            df['_tmp'] = sig.astype(int)
            r = test_signal(df, '_tmp', f'{ticker} | {name}')
            if r:
                results.append({**r, 'Ticker': ticker, 'Category': 'Composite'})

        # Category 7 — Cross-asset TNX
        if 'TNX_spike' in df.columns:
            df['_tmp'] = df['TNX_spike']
            r = test_signal(df, '_tmp', f'{ticker} | TNX_yield_spike_2d')
            if r:
                results.append({**r, 'Ticker': ticker, 'Category': 'CrossAsset'})
            if 'Down3_2d' in df.columns:
                df['_tmp'] = ((df['Down3_2d'] == 1) & (df['TNX_spike'] == 1)).astype(int)
                r = test_signal(df, '_tmp', f'{ticker} | Down2d+TNX_spike')
                if r:
                    results.append({**r, 'Ticker': ticker, 'Category': 'CrossAsset'})

    # Clean up temp column
    for df in frames.values():
        if '_tmp' in df.columns:
            df.drop(columns=['_tmp'], inplace=True)

    res = pd.DataFrame(results).dropna()
    return res.sort_values('Sharpe', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  OOS VALIDATOR
# ─────────────────────────────────────────────────────────────

def validate_oos(frames, top_signals, tp=TP_DEFAULT, sl=SL_DEFAULT):
    """Run OOS validation on top signals. Returns list of validated edges."""
    cutoff = pd.Timestamp(OOS_CUTOFF)
    validated = []

    print(f"\n{'='*65}")
    print(f"  OOS VALIDATION  (split: {OOS_CUTOFF})")
    print(f"  TP: {tp*100:.2f}%  |  SL: {sl*100:.2f}%")
    print(f"{'='*65}")

    for _, row in top_signals.iterrows():
        ticker = row['Ticker']
        if ticker not in frames:
            continue
        df = frames[ticker].copy()
        # Re-derive the signal column
        sig_label = row['Signal'].split('|')[1].strip()
        # Try to find matching column
        matching_col = None
        for col in df.columns:
            if col in sig_label or sig_label.replace('+', '_') == col:
                matching_col = col
                break
        if matching_col is None:
            continue

        trades = run_backtest(df, matching_col, tp, sl)
        if trades.empty:
            continue

        ins  = trades[trades['Sample'] == 'IN']
        oos  = trades[trades['Sample'] == 'OUT']
        if len(ins) < 5 or len(oos) < 3:
            continue

        oos_wr     = oos['Win'].mean() * 100
        oos_sharpe = (oos['PnL'].mean() / oos['PnL'].std() * np.sqrt(252)
                      if oos['PnL'].std() > 0 else 0)
        survived = oos_sharpe >= OOS_SHARPE_MIN and oos_wr >= OOS_WR_MIN

        status = '✅ VALIDATED' if survived else '❌ FAILED'
        print(f"\n  {status} | {row['Signal']}")
        print(f"    IN : N={len(ins):>3} | WR={ins['Win'].mean()*100:.1f}% | "
              f"Sharpe={ins['PnL'].mean()/ins['PnL'].std()*np.sqrt(252):.2f}")
        print(f"    OUT: N={len(oos):>3} | WR={oos_wr:.1f}% | Sharpe={oos_sharpe:.2f}")

        if survived:
            validated.append({**row.to_dict(), 'OOS_WR': oos_wr,
                               'OOS_Sharpe': oos_sharpe, 'OOS_N': len(oos)})

    return validated


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Systematic Edge Scanner')
    parser.add_argument('--spy',    required=True,  help='SPY CSV path')
    parser.add_argument('--vix',    required=True,  help='VIX CSV path')
    parser.add_argument('--qqq',    default=None,   help='QQQ CSV path')
    parser.add_argument('--iwm',    default=None,   help='IWM CSV path')
    parser.add_argument('--tnx',    default=None,   help='10Y Yield CSV path')
    parser.add_argument('--top-n',  type=int, default=20, help='Top N signals to validate OOS')
    parser.add_argument('--output', default='results/edge_scan_results.csv')
    args = parser.parse_args()

    print("\n  Loading data...")
    vix = load_vix_csv(args.vix)
    tnx = load_investing_csv(args.tnx, 'TNX') if args.tnx else None

    spy = load_spy_nasdaq_csv(args.spy, 'SPY')
    spy_f = build_features(spy, vix, tnx)
    frames = {'SPY': spy_f}

    if args.qqq:
        qqq = load_investing_csv(args.qqq, 'QQQ')
        frames['QQQ'] = build_features(qqq, vix, tnx)
    if args.iwm:
        iwm = load_investing_csv(args.iwm, 'IWM')
        frames['IWM'] = build_features(iwm, vix, tnx)

    print(f"\n  Running edge scanner across {len(frames)} tickers...")
    all_edges = run_full_scan(frames)

    print(f"\n  Signals tested:            {len(all_edges)}")
    print(f"  Significant (p < 0.05):    {(all_edges['p_val'] < 0.05).sum()}")
    print(f"\n  TOP {args.top_n} BY SHARPE:")
    print(all_edges[['Signal', 'N', 'WinRate', 'AvgReturn', 'Sharpe', 'p_val',
                      'Category']].head(args.top_n).to_string(index=False))

    top = all_edges[all_edges['p_val'] < 0.10].head(args.top_n)
    validated = validate_oos(frames, top)

    print(f"\n{'='*65}")
    print(f"  FINAL: {len(validated)} edges survived OOS validation")
    print(f"{'='*65}")
    for v in validated:
        print(f"  ★  {v['Signal']}")
        print(f"     OOS WR={v['OOS_WR']:.1f}%  Sharpe={v['OOS_Sharpe']:.2f}  N={v['OOS_N']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_edges.to_csv(args.output, index=False)
    print(f"\n  Results saved → {args.output}")


if __name__ == '__main__':
    main()
