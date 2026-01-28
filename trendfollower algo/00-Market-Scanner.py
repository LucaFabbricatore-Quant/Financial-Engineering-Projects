import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURAZIONI ---
TICKER_MAP = {
    'AAPL': 'AAPL', 'MSFT': 'MSFT', 'GOOG': 'GOOG', 'AMZN': 'AMZN',
    'META': 'META', 'NVDA': 'NVDA', 'NFLX': 'NFLX', 'GC=F': 'XAUUSD'
}

# Struttura dati coerente con il tuo primo script di debug
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

# --- FUNZIONI MATEMATICHE PURE (IDENTICHE AL TUO TEMPLATE) ---

def get_rsi_wilder(series, period=28):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def get_kama(series, period):
    change = abs(series - series.shift(period))
    volatility = abs(series - series.shift(1)).rolling(period).sum()
    er = change / volatility
    sc = (er * (2/3 - 2/31) + 2/31)**2
    kama = series.copy()
    kama.iloc[period-1] = series.iloc[:period].mean() 
    for i in range(period, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
    return kama

def get_tech_series(ticker, df):
    """Calcola la serie dei segnali TECH (1 o 0) allineata al debug"""
    p = CONFIG_ASSET[ticker]
    close = df['close']
    
    if p['tipo'] in ['VETO', 'GOLD']:
        # Calcolo EMAs
        e1 = close.ewm(span=p['ema_p'][0], adjust=False).mean()
        e2 = close.ewm(span=p['ema_p'][1], adjust=False).mean()
        e3 = close.ewm(span=p['ema_p'][2], adjust=False).mean()
        
        # Calcolo MACD
        mf = close.ewm(span=p['macd_p'][0], adjust=False).mean()
        ms = close.ewm(span=p['macd_p'][1], adjust=False).mean()
        ml = mf - ms
        sl = ml.ewm(span=p['macd_p'][2], adjust=False).mean()
        
        # LOGICA AGGIORNATA: Inserito e2 nel controllo del trend
        # Il segnale è OK se la gerarchia delle medie è rispettata (e1 > e2 > e3) E il MACD è sopra il signal
        return ((e1 > e2) & (e2 > e3) & (ml > sl)).astype(int)

    elif p['tipo'] == 'KAMA':
        kf = get_kama(close, p['fast'])
        ks = get_kama(close, p['slow'])
        return (kf > ks).astype(int)

    elif p['tipo'] == 'AAPL_SPECIAL':
        ema_f = close.ewm(span=p['macd']['fast'], adjust=False).mean()
        ema_s = close.ewm(span=p['macd']['slow'], adjust=False).mean()
        macd_l = ema_f - ema_s
        macd_s = macd_l.ewm(span=p['macd']['signal'], adjust=False).mean()
        sma = close.rolling(p['bb']['window']).mean()
        std = close.rolling(p['bb']['window']).std()
        low = sma - (std * p['bb']['sigma'])
        return ((macd_l > macd_s) | (close < low)).astype(int)

    return pd.Series(0, index=df.index)

# --- DASHBOARD ENGINE ---

def run_market_scanner():
    if not mt5.initialize(): quit()
    
    print(f"\n📊 MARKET DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*115)
    print(f"{'ASSET':<8} | {'PRICE':<10} | {'TECH':<5} | {'RSI(28)':<8} | {'TREND':<10} | {'DAYS':<6} | {'LOGIC'}")
    print("-" * 115)

    for ticker, mt5_s in TICKER_MAP.items():
        r = mt5.copy_rates_from_pos(mt5_s, mt5.TIMEFRAME_D1, 0, 1000)
        if r is None: continue
        df = pd.DataFrame(r)
        df['close'] = df['close'] # Assicuriamoci che la colonna esista per le funzioni
        
        # Calcolo serie storiche
        tech_series = get_tech_series(ticker, df)
        rsi_series = get_rsi_wilder(df['close'], 28)
        
        # Stato corrente
        current_tech = tech_series.iloc[-1]
        current_rsi = rsi_series.iloc[-1]
        is_long = (current_tech == 1 and current_rsi > 50)
        
        # Calcolo Giorni Consecutivi (Trend Invariato)
        status_history = (tech_series == 1) & (rsi_series > 50)
        consecutive_days = 0
        for val in reversed(status_history.values[:-1]):
            if val == is_long:
                consecutive_days += 1
            else:
                break
        
        # Output
        price = df['close'].iloc[-1]
        trend_str = "🟩 LONG" if is_long else "⬜ FLAT"
        emoji_rsi = "🔥" if current_rsi > 65 else "❄️" if current_rsi < 40 else "  "
        tech_display = " 1 " if current_tech == 1 else " 0 "

        print(f"{ticker:<8} | {price:<10.2f} | {tech_display:<5} | {current_rsi:<6.2f} {emoji_rsi} | {trend_str:<10} | {consecutive_days:<6} | {CONFIG_ASSET[ticker]['tipo']}")

    print("-" * 115)
    print("Logica VETO/GOLD: (E1 > E2 > E3) AND (MACD_L > MACD_S)")
    mt5.shutdown()

if __name__ == "__main__":
    run_market_scanner()