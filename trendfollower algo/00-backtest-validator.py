import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# 1. CONFIGURAZIONE
# ============================================================
TICKER_MAP = {
    'AAPL': 'AAPL', 'MSFT': 'MSFT', 'GOOG': 'GOOG', 'AMZN': 'AMZN',
    'META': 'META', 'NVDA': 'NVDA', 'NFLX': 'NFLX', 'GC=F': 'XAUUSD'
}

CONFIG_ASSET = {
    'AMZN': {'tipo': 'KAMA', 'fast': 32, 'slow': 65},
    'META': {'tipo': 'KAMA', 'fast': 31, 'slow': 70},
    'MSFT': {'tipo': 'KAMA', 'fast': 34, 'slow': 65},
    'NFLX': {'tipo': 'KAMA', 'fast': 12, 'slow': 57},
    'AAPL': {'tipo': 'AAPL_SPECIAL', 'bb': {'window': 10, 'sigma': 2.3}, 'macd': {'fast': 36, 'slow': 52, 'signal': 29}},
    'GOOG': {'tipo': 'VETO', 'ema_p': [8, 72, 230], 'macd_p': [44, 45, 33]},
    'NVDA': {'tipo': 'VETO', 'ema_p': [8, 56, 165], 'macd_p': [34, 88, 6]},
    'GC=F': {'tipo': 'GOLD', 'ema_p': [6, 181, 85], 'macd_p': [48, 59, 50]}
}

# ============================================================
# 2. INDICATORI (IDENTICI AL LIVE)
# ============================================================
def get_rsi_wilder(series, period=28):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def get_kama(series, period):
    change = abs(series - series.shift(period))
    volatility = abs(series - series.shift(1)).rolling(period).sum()
    er = change / volatility
    sc = (er * (2/3 - 2/31) + 2/31)**2
    kama = series.copy()
    for i in range(period, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
    return kama

def get_macd(series, fast, slow, signal):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line

def get_bb(series, window, sigma):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + (std * sigma), sma, sma - (std * sigma)

# ============================================================
# 3. LOGICHE DI STATO E SELEZIONE
# ============================================================
def evaluate_status(ticker, series, prev_pos, states_mem):
    conf = CONFIG_ASSET[ticker]
    if conf['tipo'] == 'KAMA':
        kf, ks = get_kama(series, conf['fast']), get_kama(series, conf['slow'])
        return 1 if kf.iloc[-1] > ks.iloc[-1] else 0
    elif conf['tipo'] == 'AAPL_SPECIAL':
        macd, sig = get_macd(series, conf['macd']['fast'], conf['macd']['slow'], conf['macd']['signal'])
        _, mid, low = get_bb(series, conf['bb']['window'], conf['bb']['sigma'])
        return 1 if (macd.iloc[-1] > sig.iloc[-1] or series.iloc[-1] < low.iloc[-1]) else 0
    elif conf['tipo'] in ['VETO', 'GOLD']:
        e = conf['ema_p']; m_p = conf['macd_p']
        e1, e3 = series.ewm(span=e[0]).mean(), series.ewm(span=e[2]).mean()
        m, s = get_macd(series, m_p[0], m_p[1], m_p[2])
        return 1 if (e1.iloc[-1] > e3.iloc[-1] and m.iloc[-1] > s.iloc[-1]) else 0
    return 0

def build_selection(window_df):
    """Replica Step 2 Notebook: Rank + Safe Haven"""
    eligible = []
    for t in TICKER_MAP.keys():
        if t == 'GC=F': continue
        st = evaluate_status(t, window_df[t], 0, {})
        rsi = get_rsi_wilder(window_df[t], 28).iloc[-1]
        if st == 1 and rsi > 50:
            eligible.append((t, rsi))
    
    # Ranking
    eligible.sort(key=lambda x: x[1], reverse=True)
    selected = [x[0] for x in eligible[:4]]
    
    # Safe Haven filling
    slots_gap = 4 - len(selected)
    if slots_gap > 0:
        gold_st = evaluate_status('GC=F', window_df['GC=F'], 0, {})
        gold_rsi = get_rsi_wilder(window_df['GC=F'], 28).iloc[-1]
        if gold_st == 1 and gold_rsi > 50:
            selected.extend(['GC=F'] * slots_gap)
    return selected

# ============================================================
# 4. LOOP DI BACKTEST AVANZATO
# ============================================================
def run_backtest():
    if not mt5.initialize(): quit()
    
    # Download dati (1000 giorni per stabilità)
    data = {}
    for disp, mt5_s in TICKER_MAP.items():
        mt5.symbol_select(mt5_s, True)
        r = mt5.copy_rates_from_pos(mt5_s, mt5.TIMEFRAME_D1, 0, 1000)
        df_t = pd.DataFrame(r)
        df_t.columns = [c.lower() for c in df_t.columns]
        df_t['time'] = pd.to_datetime(df_t['time'], unit='s')
        data[disp] = df_t.set_index('time')['close']
    
    prices = pd.DataFrame(data).dropna()
    returns_df = prices.pct_change() # Ritorni giornalieri singoli asset
    
    history = []
    prev_portfolio = []
    daily_returns = []

    # Inizio simulazione (dopo 500 giorni di warm-up)
    start_idx = 500
    for i in range(start_idx, len(prices)):
        today_date = prices.index[i]
        
        # 1. Determiniamo la selezione di IERI (-1) e L'ALTRO IERI (-2) per decidere OGGI
        # Questo replica il "Lag 1-day" del notebook
        cand_yesterday = build_selection(prices.iloc[:i]) 
        cand_day_before = build_selection(prices.iloc[:i-1])

        # 2. Logica di Stabilità di Blocco
        if sorted(cand_yesterday) == sorted(cand_day_before):
            current_portfolio = cand_yesterday
        else:
            # Instabilità: tieni i precedenti se ancora validi oggi
            valid_now = []
            for asset in prev_portfolio:
                st = evaluate_status(asset, prices.iloc[:i+1][asset], 0, {})
                rsi = get_rsi_wilder(prices.iloc[:i+1][asset], 28).iloc[-1]
                if st == 1 and rsi > 50: valid_now.append(asset)
            
            # Se mancano pezzi, riempi con Oro fresco
            gap = 4 - len(valid_now)
            if gap > 0:
                gold_st = evaluate_status('GC=F', prices.iloc[:i+1]['GC=F'], 0, {})
                gold_rsi = get_rsi_wilder(prices.iloc[:i+1]['GC=F'], 28).iloc[-1]
                if gold_st == 1 and gold_rsi > 50: valid_now.extend(['GC=F'] * gap)
            current_portfolio = valid_now

        # 3. Calcolo Ritorno del Portafoglio per OGGI
        if current_portfolio:
            # Ogni slot è il 25% del capitale (Molt. 0.75 gestito dopo nel capitale rif)
            # Calcoliamo il ritorno medio degli asset in portafoglio
            rets = [returns_df.loc[today_date, asset] for asset in current_portfolio]
            day_ret = np.mean(rets)
        else:
            day_ret = 0
            
        daily_returns.append(day_ret)
        prev_portfolio = current_portfolio.copy()
        history.append({'Data': today_date, 'Assets': ", ".join(current_portfolio)})

    # ============================================================
    # 5. CALCOLO METRICHE (CONFRONTO IMMAGINE)
    # ============================================================
    rets_series = pd.Series(daily_returns)
    equity_curve = (1 + rets_series).cumprod()
    
    # Metriche LTM (Last 252 days)
    ltm_rets = rets_series.tail(252)
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[-252]) - 1
    
    # Drawdown
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve / cum_max) - 1
    max_dd = drawdown.tail(252).min()
    
    # Volatilità e Sharpe
    vol = ltm_rets.std() * np.sqrt(252)
    sharpe = (ltm_rets.mean() * 252) / vol if vol > 0 else 0
    
    # Trades e Segments (Cambi di portafoglio)
    segments = 0
    for j in range(1, len(history)):
        if history[j]['Assets'] != history[j-1]['Assets']: segments += 1

    print("\n" + "="*40)
    print("📊 WINDOW METRICS (LTM)")
    print("="*40)
    print(f"Period: {prices.index[-252].date()} to {prices.index[-1].date()}")
    print(f"Return:      {total_return:+.2%}")
    print(f"Max DD:      {max_dd:.2%}")
    print(f"Volatility:  {vol:.1%}")
    print(f"Sharpe:      {sharpe:.2f}")
    print(f"Segments:    {segments}")
    print("="*40)

    # Salvataggio
    pd.DataFrame(history).to_csv("backtest_stability_report.csv", index=False)
    print("\n✅ Report dettagliato salvato in 'backtest_stability_report.csv'")

run_backtest()