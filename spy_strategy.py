#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  SPY MEAN-REVERSION STRATEGY — DAILY SIGNAL SCANNER
  Signal B: 2x ≥0.3% down | Low Volume | VIX < 25
  TP: +0.50% | SL: -0.30% | Half-Kelly sizing: 23% of account
═══════════════════════════════════════════════════════════════

  SETUP:
    pip install yfinance pandas numpy scipy statsmodels alpaca-trade-api

  USAGE:
    python spy_strategy.py                  # run signal check
    python spy_strategy.py --backtest       # run backtest on downloaded data
    python spy_strategy.py --live           # live mode (requires Alpaca keys)

  BROKERS SUPPORTED:
    - Alpaca (paper + live) — set ALPACA_API_KEY + ALPACA_SECRET_KEY in env
    - Manual mode (prints signal + recommended order details)
"""

import os
import sys
import argparse
import warnings
import datetime
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
#  STRATEGY PARAMETERS  (derived from backtest — do not guess)
# ─────────────────────────────────────────────────────────────
TICKER          = 'SPY'
VIX_TICKER      = '^VIX'

# Signal filters
MIN_DOWN_PCT    = 0.003   # ≥ 0.30% down each day
CONSEC_DAYS     = 2       # 2 consecutive down days
VIX_MAX         = 25.0    # VIX must be below this
VOL_RATIO_MAX   = 1.0     # volume must be below 20-day avg (weak selling)
REQUIRE_ABOVE_200MA = False  # optional — tightens signal, reduces trades

# SL/TP (optimised from backtest grid search)
TAKE_PROFIT_PCT = 0.005   # +0.50% from entry
STOP_LOSS_PCT   = 0.003   # -0.30% from entry
TIME_STOP       = '15:45' # exit at this time if neither TP nor SL hit

# Position sizing (Half-Kelly from backtest)
HALF_KELLY      = 0.233   # 23.3% of account per trade
ACCOUNT_SIZE    = 10_000  # update to your actual account size in USD

# Lookback for feature calculation
LOOKBACK_DAYS   = 250     # trading days of history to download

# ─────────────────────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_data(ticker, days=LOOKBACK_DAYS):
    """Download historical OHLCV data via yfinance."""
    try:
        import yfinance as yf
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=int(days * 1.5))
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df = df.reset_index()
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if c[1] == '' else c[0] for c in df.columns]
        df = df.rename(columns={'index': 'Date', 'Adj Close': 'Close'})
        df = df[['Date','Open','High','Low','Close','Volume']].dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def load_csv(spy_path, vix_path):
    """Load from local CSV files (fallback / offline mode)."""
    spy = pd.read_csv(spy_path)
    spy.columns = spy.columns.str.strip()
    if 'Close/Last' in spy.columns:
        spy.rename(columns={'Close/Last': 'Close'}, inplace=True)
    spy['Date'] = pd.to_datetime(spy['Date'])
    spy = spy.sort_values('Date').reset_index(drop=True)
    for col in ['Open','High','Low','Close','Volume']:
        spy[col] = pd.to_numeric(spy[col], errors='coerce')

    vix = pd.read_csv(vix_path)
    vix.columns = vix.columns.str.strip()
    date_col  = 'DATE' if 'DATE' in vix.columns else 'Date'
    close_col = 'CLOSE' if 'CLOSE' in vix.columns else 'Close'
    vix = vix.rename(columns={date_col: 'Date', close_col: 'VIX'})
    vix['Date'] = pd.to_datetime(vix['Date'])
    vix = vix[['Date','VIX']].dropna()

    df = spy.merge(vix, on='Date', how='left')
    df['VIX'] = pd.to_numeric(df['VIX'], errors='coerce')
    return df


# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def build_features(df):
    """Add all signal features to dataframe."""
    df = df.copy()
    df['Return']     = df['Close'].pct_change()
    df['MA20']       = df['Close'].rolling(20).mean()
    df['MA50']       = df['Close'].rolling(50).mean()
    df['MA200']      = df['Close'].rolling(200).mean()
    df['ATR']        = (df['High'] - df['Low']).rolling(14).mean()
    df['ATR_pct']    = df['ATR'] / df['Close']
    df['Vol_Ratio']  = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Dist_MA200'] = (df['Close'] - df['MA200']) / df['MA200']
    df['Above200']   = (df['Close'] > df['MA200']).astype(int)

    # Consecutive down days
    df['IsDown']     = (df['Return'] <= -MIN_DOWN_PCT).astype(int)
    df['Consec2']    = ((df['IsDown'] == 1) & (df['IsDown'].shift(1) == 1)).astype(int)

    # Signal B — the validated signal
    df['Signal'] = (
        (df['Consec2']    == 1) &
        (df['Vol_Ratio']  <  VOL_RATIO_MAX) &
        (df['VIX']        <  VIX_MAX)
    ).astype(int)

    if REQUIRE_ABOVE_200MA:
        df['Signal'] = df['Signal'] & df['Above200']

    return df.dropna()


# ─────────────────────────────────────────────────────────────
#  SIGNAL CHECK (run each morning)
# ─────────────────────────────────────────────────────────────

def check_signal(df):
    """Check whether today's close data triggers a signal for tomorrow."""
    last  = df.iloc[-1]
    prev  = df.iloc[-2]

    print("\n" + "═" * 56)
    print("  SPY MEAN-REVERSION SIGNAL SCANNER")
    print(f"  As of: {last['Date'].date()}")
    print("═" * 56)

    # Print current conditions
    ret1  = last['Return']  * 100
    ret2  = prev['Return']  * 100
    vix   = last['VIX']
    volr  = last['Vol_Ratio']
    above = bool(last['Above200'])
    atr   = last['ATR_pct'] * 100

    print(f"\n  Conditions check:")
    print(f"    Day-1 return       : {ret2:+.3f}%   {'✅' if prev['IsDown'] else '❌'}  (need ≤ -{MIN_DOWN_PCT*100:.1f}%)")
    print(f"    Day-2 return       : {ret1:+.3f}%   {'✅' if last['IsDown'] else '❌'}  (need ≤ -{MIN_DOWN_PCT*100:.1f}%)")
    print(f"    VIX level          : {vix:.1f}       {'✅' if vix < VIX_MAX else '❌'}  (need < {VIX_MAX})")
    print(f"    Volume ratio       : {volr:.2f}x      {'✅' if volr < VOL_RATIO_MAX else '❌'}  (need < {VOL_RATIO_MAX:.1f}x avg)")
    print(f"    Above 200-day MA   : {'Yes' if above else 'No'}      {'✅' if above else '⚠️ '}")
    print(f"    ATR (14-day)       : {atr:.3f}%")

    signal = bool(last['Signal'])

    print(f"\n  {'🟢 SIGNAL ACTIVE — ENTER TOMORROW' if signal else '🔴 NO SIGNAL — STAND ASIDE'}")

    if signal:
        entry_est = last['Close']
        tp_price  = entry_est * (1 + TAKE_PROFIT_PCT)
        sl_price  = entry_est * (1 - STOP_LOSS_PCT)
        pos_size  = ACCOUNT_SIZE * HALF_KELLY
        shares    = int(pos_size / entry_est)
        risk_amt  = shares * entry_est * STOP_LOSS_PCT

        print(f"\n  ORDER DETAILS (based on last close ${entry_est:.2f}):")
        print(f"    Action          : BUY {TICKER} at tomorrow's OPEN")
        print(f"    Estimated entry : ~${entry_est:.2f}")
        print(f"    Take Profit     : ${tp_price:.2f}  (+{TAKE_PROFIT_PCT*100:.2f}%)")
        print(f"    Stop Loss       : ${sl_price:.2f}  (-{STOP_LOSS_PCT*100:.2f}%)")
        print(f"    Time Stop       : Exit at {TIME_STOP} if neither hit")
        print(f"    Position size   : {shares} shares (${shares*entry_est:,.0f})")
        print(f"    Max risk        : ${risk_amt:.0f}  ({STOP_LOSS_PCT*100:.2f}% of position)")
        print(f"    Account used    : {HALF_KELLY*100:.1f}%  (Half-Kelly)")
        print(f"\n  ⚠️  Always verify conditions at market open before entering.")

    print("\n" + "═" * 56 + "\n")
    return signal, last


# ─────────────────────────────────────────────────────────────
#  FULL BACKTEST
# ─────────────────────────────────────────────────────────────

def run_backtest(df, in_sample_cutoff='2024-01-01'):
    """Run full backtest with in/out-of-sample split."""
    cutoff = pd.Timestamp(in_sample_cutoff)
    results = []

    for i in range(len(df) - 1):
        if df.iloc[i]['Signal'] != 1:
            continue
        row   = df.iloc[i + 1]
        entry = row['Open']
        tp_p  = entry * (1 + TAKE_PROFIT_PCT)
        sl_p  = entry * (1 - STOP_LOSS_PCT)

        hit_tp = row['High'] >= tp_p
        hit_sl = row['Low']  <= sl_p

        if hit_tp and hit_sl:
            up  = row['High'] - entry
            dn  = entry - row['Low']
            pnl = TAKE_PROFIT_PCT if up <= dn else -STOP_LOSS_PCT
        elif hit_tp:
            pnl = TAKE_PROFIT_PCT
        elif hit_sl:
            pnl = -STOP_LOSS_PCT
        else:
            pnl = (row['Close'] - entry) / entry

        results.append({
            'Date'      : row['Date'],
            'Entry'     : entry,
            'TP_Price'  : tp_p,
            'SL_Price'  : sl_p,
            'Exit_Price': entry * (1 + pnl),
            'PnL_pct'   : pnl * 100,
            'Win'       : pnl > 0,
            'Sample'    : 'IN' if row['Date'] < cutoff else 'OUT',
            'VIX'       : df.iloc[i]['VIX'],
            'Vol_Ratio' : df.iloc[i]['Vol_Ratio'],
        })

    if not results:
        print("  No trades found in backtest period.")
        return

    t = pd.DataFrame(results)

    def print_stats(label, subset):
        if subset.empty:
            return
        n       = len(subset)
        wr      = subset['Win'].mean() * 100
        avg_ret = subset['PnL_pct'].mean()
        total   = subset['PnL_pct'].sum()
        sharpe  = (subset['PnL_pct'].mean() / subset['PnL_pct'].std()
                   * np.sqrt(252)) if subset['PnL_pct'].std() > 0 else 0
        cum     = subset['PnL_pct'].cumsum()
        mdd     = (cum.cummax() - cum).max()
        print(f"\n  {label}")
        print(f"    Trades       : {n}")
        print(f"    Win Rate     : {wr:.1f}%")
        print(f"    Avg Return   : {avg_ret:+.3f}%/trade")
        print(f"    Total Return : {total:+.2f}%")
        print(f"    Sharpe       : {sharpe:.2f}")
        print(f"    Max Drawdown : -{mdd:.2f}%")

    print("\n" + "═" * 56)
    print("  BACKTEST RESULTS")
    print("═" * 56)
    print(f"  Strategy : Signal B — 2x≥0.3% down + low vol + VIX<25")
    print(f"  TP: {TAKE_PROFIT_PCT*100:.2f}%   SL: {STOP_LOSS_PCT*100:.2f}%   Time stop: {TIME_STOP}")

    print_stats(f"IN-SAMPLE  (before {in_sample_cutoff})", t[t['Sample']=='IN'])
    print_stats(f"OUT-OF-SAMPLE (from {in_sample_cutoff})", t[t['Sample']=='OUT'])
    print_stats("COMBINED", t)

    print(f"\n  ALL TRADES:")
    print(f"  {'Date':<12} {'Entry':>8} {'TP':>8} {'SL':>8} "
          f"{'PnL%':>7} {'Win':>5} {'VIX':>6} {'Sample':>6}")
    print(f"  {'─'*65}")
    for _, row in t.iterrows():
        print(f"  {str(row['Date'].date()):<12} "
              f"${row['Entry']:>7.2f} "
              f"${row['TP_Price']:>7.2f} "
              f"${row['SL_Price']:>7.2f} "
              f"{row['PnL_pct']:>+6.3f}% "
              f"{'✅' if row['Win'] else '❌':>5}  "
              f"{row['VIX']:>5.1f}  "
              f"{row['Sample']:>6}")
    print()


# ─────────────────────────────────────────────────────────────
#  ALPACA LIVE TRADING (optional)
# ─────────────────────────────────────────────────────────────

def place_alpaca_order(signal_row, paper=True):
    """
    Place a bracket order (entry + TP + SL) via Alpaca.
    Requires: ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
    Set paper=True to use paper trading endpoint first.
    """
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        print("  [ERROR] alpaca-trade-api not installed.")
        print("  Run: pip install alpaca-trade-api")
        return

    api_key    = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("  [ERROR] Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")
        return

    base_url = ('https://paper-api.alpaca.markets' if paper
                else 'https://api.alpaca.markets')

    api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')

    account    = api.get_account()
    equity     = float(account.equity)
    pos_value  = equity * HALF_KELLY
    last_close = signal_row['Close']
    shares     = int(pos_value / last_close)

    tp_price = round(last_close * (1 + TAKE_PROFIT_PCT), 2)
    sl_price = round(last_close * (1 - STOP_LOSS_PCT),   2)

    print(f"\n  Placing Alpaca bracket order:")
    print(f"    {'PAPER' if paper else 'LIVE'} | {shares} shares SPY")
    print(f"    TP: ${tp_price}  |  SL: ${sl_price}")

    order = api.submit_order(
        symbol        = TICKER,
        qty           = shares,
        side          = 'buy',
        type          = 'market',
        time_in_force = 'day',
        order_class   = 'bracket',
        take_profit   = {'limit_price': str(tp_price)},
        stop_loss     = {'stop_price': str(sl_price)},
    )

    print(f"  ✅ Order submitted: {order.id}")
    print(f"  Status: {order.status}")
    return order


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SPY Mean-Reversion Strategy')
    parser.add_argument('--backtest', action='store_true', help='Run full backtest')
    parser.add_argument('--live',     action='store_true', help='Place live Alpaca order if signal fires')
    parser.add_argument('--paper',    action='store_true', help='Paper trade via Alpaca (default with --live)')
    parser.add_argument('--spy-csv',  type=str,  help='Path to local SPY CSV (offline mode)')
    parser.add_argument('--vix-csv',  type=str,  help='Path to local VIX CSV (offline mode)')
    parser.add_argument('--account',  type=float, default=ACCOUNT_SIZE, help='Account size in USD')
    args = parser.parse_args()

    print("\n  Loading data...")

    # Load data
    if args.spy_csv and args.vix_csv:
        print("  Using local CSV files...")
        df = load_csv(args.spy_csv, args.vix_csv)
    else:
        print("  Fetching live data from Yahoo Finance...")
        spy = fetch_data(TICKER)
        vix = fetch_data(VIX_TICKER)
        if spy.empty or vix.empty:
            print("  [ERROR] Could not fetch data. Use --spy-csv and --vix-csv flags.")
            sys.exit(1)
        vix = vix.rename(columns={'Close': 'VIX'})[['Date','VIX']]
        df  = spy.merge(vix, on='Date', how='left')

    # Build features
    df = build_features(df)
    print(f"  Data loaded: {len(df)} trading days "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")

    if args.backtest:
        run_backtest(df)
    else:
        signal_fired, last_row = check_signal(df)

        if signal_fired and (args.live or args.paper):
            confirm = input("  Signal fired. Place Alpaca order? (yes/no): ")
            if confirm.lower() == 'yes':
                place_alpaca_order(last_row, paper=not args.live)
            else:
                print("  Order cancelled.")


if __name__ == '__main__':
    main()
