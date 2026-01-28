import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================
CAPITALE_RIF = 10000
MOLTIPLICATORE = 0.75
STATE_FILE = "trading_state.json"

TICKER_MAP = {
    'AAPL': 'AAPL', 'MSFT': 'MSFT', 'GOOG': 'GOOG', 'AMZN': 'AMZN',
    'META': 'META', 'NVDA': 'NVDA', 'NFLX': 'NFLX', 'GC=F': 'XAUUSD'
}

CONFIG_ASSET = {
    "AAPL": {"tipo": "AAPL_SPECIAL", "bb": {"window": 10, "sigma": 2.3}, "macd": {"fast": 36, "slow": 52, "signal": 29}},
    "GOOG": {"tipo": "VETO", "ema_p": [8, 72, 230], "macd_p": [44, 45, 33]},
    "NVDA": {"tipo": "VETO", "ema_p": [8, 56, 165], "macd_p": [34, 88, 6]},
    "GC=F": {"tipo": "GOLD", "ema_p": [6, 181, 85], "macd_p": [48, 59, 50]},
    "MSFT": {"tipo": "KAMA", "fast": 34, "slow": 65},
    "AMZN": {"tipo": "KAMA", "fast": 32, "slow": 65},
    "META": {"tipo": "KAMA", "fast": 31, "slow": 70},
    "NFLX": {"tipo": "KAMA", "fast": 12, "slow": 57}
}

# --- FUNZIONI MATEMATICHE ---
def get_rsi_wilder(series, period=28):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

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

# ============================================================
# 3. VALUTAZIONE ASSET (LOGICA STATI TRIPLE EMA)
# ============================================================
def evaluate_asset_status(ticker, df, idx, mem):
    is_active = ticker in mem.get('active_positions_list', [])
    series = df[ticker].iloc[:len(df) + idx + 1] if idx < 0 else df[ticker]
    p = CONFIG_ASSET[ticker]
    
    if p['tipo'] in ['VETO', 'GOLD']:
        # 1. Indicatori
        e1 = series.ewm(span=p['ema_p'][0], adjust=False).mean()
        e2 = series.ewm(span=p['ema_p'][1], adjust=False).mean()
        e3 = series.ewm(span=p['ema_p'][2], adjust=False).mean()
        
        m_l = series.ewm(span=p['macd_p'][0], adjust=False).mean() - \
              series.ewm(span=p['macd_p'][1], adjust=False).mean()
        s_l = m_l.ewm(span=p['macd_p'][2], adjust=False).mean()

        state_pos = 0
        # Aumentato a 250 per convergenza EMA lunghe
        start_sim = max(1, len(series) - 250)
        
        for i in range(start_sim, len(series)):
            # --- Rilevazione Incroci (Includendo 2 vs 3) ---
            ema_up = (e1.iloc[i-1] <= e2.iloc[i-1] and e1.iloc[i] > e2.iloc[i]) or \
                     (e1.iloc[i-1] <= e3.iloc[i-1] and e1.iloc[i] > e3.iloc[i]) or \
                     (e2.iloc[i-1] <= e3.iloc[i-1] and e2.iloc[i] > e3.iloc[i]) # <--- Aggiunto
            
            ema_down = (e1.iloc[i-1] >= e2.iloc[i-1] and e1.iloc[i] < e2.iloc[i]) or \
                       (e1.iloc[i-1] >= e3.iloc[i-1] and e1.iloc[i] < e3.iloc[i]) or \
                       (e2.iloc[i-1] >= e3.iloc[i-1] and e2.iloc[i] < e3.iloc[i]) # <--- Aggiunto
            
            macd_up = (m_l.iloc[i-1] <= s_l.iloc[i-1] and m_l.iloc[i] > s_l.iloc[i])
            macd_down = (m_l.iloc[i-1] >= s_l.iloc[i-1] and m_l.iloc[i] < s_l.iloc[i])

            s1 = 1 if ema_up else (-1 if ema_down else 0)
            s2 = 1 if macd_up else (-1 if macd_down else 0)

            # --- LOGICA ENSEMBLE ---
            if p['tipo'] == 'VETO':
                if state_pos == 0:
                    # Entrata restrittiva: nessuno dei due deve essere "contro"
                    if (s1 == 1 and s2 != -1) or (s2 == 1 and s1 != -1): state_pos = 1
                else:
                    # Uscita immediata: basta un incrocio down di uno dei due
                    if s1 == -1 or s2 == -1: state_pos = 0
            
            elif p['tipo'] == 'GOLD':
                if state_pos == 0:
                    # Logica Pine Script originale: 3EMA ha priorità
                    if s1 == 1 or (s2 == 1 and s1 != -1): state_pos = 1
                else:
                    # Logica Pine Script originale: esce solo se non c'è priorità contraria
                    if s1 == -1 or (s2 == -1 and s1 != 1): state_pos = 0

        return state_pos

    # --- LOGICA KAMA & AAPL_SPECIAL (Invariate per stabilità numerica) ---
    elif p['tipo'] == 'KAMA':
        kf = get_kama(series, p['fast']).iloc[-1]
        ks = get_kama(series, p['slow']).iloc[-1]
        return 1 if kf > ks else 0

    elif p['tipo'] == 'AAPL_SPECIAL':
        ema_f = series.ewm(span=p['macd']['fast'], adjust=False).mean()
        ema_s = series.ewm(span=p['macd']['slow'], adjust=False).mean()
        macd_v = (ema_f - ema_s).iloc[-1]
        sig_v = (ema_f - ema_s).ewm(span=p['macd']['signal'], adjust=False).mean().iloc[-1]
        sma = series.rolling(p['bb']['window']).mean().iloc[-1]
        std = series.rolling(p['bb']['window']).std().iloc[-1]
        low = sma - (std * p['bb']['sigma'])
        if not is_active: return 1 if (macd_v > sig_v or series.iloc[-1] < low) else 0
        else:
            pc = mem.get('active_prices', {}).get(ticker)
            if pc and ((series.iloc[-1] - pc) / pc) <= -0.05: return 0
            return 0 if (macd_v <= sig_v and series.iloc[-1] > sma) else 1
    return 0

# ============================================================
# 4. ENGINE DI SELEZIONE E DIAGNOSTICA
# ============================================================
def build_selection_notebook(df, mem):
    eligible_assets = []
    active_now = mem.get('active_positions_list', [])

    print(f"\n{'ASSET':<8} | {'T-1 (T/RSI)':<15} | {'T-2 (T/RSI)':<15} | {'STABILE':<8} | {'ESITO'}")
    print("-" * 80)

    for t in TICKER_MAP.keys():
        pos_t1 = evaluate_asset_status(t, df, -1, mem)
        rsi_t1 = round(get_rsi_wilder(df[t], 28).iloc[-1], 2)
        pos_t2 = evaluate_asset_status(t, df, -2, mem)
        rsi_t2 = round(get_rsi_wilder(df[t].iloc[:-1], 28).iloc[-1], 2)
        
        # Filtro stabilità (AMZN ora passa con 50.00)
        is_stable = (pos_t1 == 1 and rsi_t1 >= 50.0) and (pos_t2 == 1 and rsi_t2 >= 50.0)
        
        if t in active_now:
            outcome = "KEEP" if pos_t1 == 1 else "EXIT"
            if pos_t1 == 1: eligible_assets.append((t, rsi_t1))
        else:
            outcome = "ENTRY" if is_stable else "SKIP"
            if is_stable: eligible_assets.append((t, rsi_t1))
        
        if t != 'GC=F':
            print(f"{t:<8} | {pos_t1} / {rsi_t1:<6.2f}    | {pos_t2} / {rsi_t2:<6.2f}    | {'SI' if is_stable else 'NO':<8} | {outcome}")

    # Diagnostica ORO
    g_p1 = evaluate_asset_status('GC=F', df, -1, mem)
    g_r1 = round(get_rsi_wilder(df['GC=F'], 28).iloc[-1], 2)
    g_p2 = evaluate_asset_status('GC=F', df, -2, mem)
    g_r2 = round(get_rsi_wilder(df['GC=F'].iloc[:-1], 28).iloc[-1], 2)
    g_stable = (g_p1 == 1 and g_r1 >= 50.0) and (g_p2 == 1 and g_r2 >= 50.0)
    g_out = "KEEP" if ('GC=F' in active_now and g_p1 == 1) else ("ENTRY" if g_stable else "SKIP")
    print(f"{'GC=F':<8} | {g_p1} / {g_r1:<6.2f}    | {g_p2} / {g_r2:<6.2f}    | {'SI' if g_stable else 'NO':<8} | {g_out}")
    print("-" * 80)

    eligible_assets.sort(key=lambda x: x[1], reverse=True)
    selected = [item[0] for item in eligible_assets[:4]]
    
    slots_needed = 4 - len(selected)
    if slots_needed > 0:
        if 'GC=F' in active_now and g_p1 == 1: selected.extend(['GC=F'] * slots_needed)
        elif g_stable: selected.extend(['GC=F'] * slots_needed)
    return selected

# ============================================================
# 5. CALCOLO LOTTI E ORDINI
# ============================================================
def calcola_lotti_finali(selected_list, df):
    if not selected_list: return {}
    capitale_op = CAPITALE_RIF * MOLTIPLICATORE
    budget_slot = capitale_op / 4
    counts = pd.Series(selected_list).value_counts()
    lotti = {}
    leftover = 0
    
    for ticker, n_slots in counts.items():
        if ticker == 'GC=F': continue
        price = df[ticker].iloc[-1]
        target_val = budget_slot * n_slots
        unita = int(target_val // price)
        if unita > 0:
            lotti[ticker] = unita
            leftover += (target_val - (unita * price))
        else:
            leftover += target_val

    if 'GC=F' in counts:
        p_gold = df['GC=F'].iloc[-1]
        t_gold = (budget_slot * counts['GC=F']) + leftover
        # Conversione Gold (1 lotto = 100 once)
        lotti['GC=F'] = round(np.floor((t_gold / p_gold) / 100 / 0.01) * 0.01, 2)
            
    return lotti

# ============================================================
# 6. ESECUZIONE (RUN ASSISTANT)
# ============================================================
def run_assistant():
    if not mt5.initialize(): 
        print("Errore MT5"); return
    
    # Caricamento memoria
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: mem = json.load(f)
    else:
        mem = {"active_positions_list": [], "active_prices": {}}

    # Scarico dati
    prices = {}
    for ticker, mt5_s in TICKER_MAP.items():
        r = mt5.copy_rates_from_pos(mt5_s, mt5.TIMEFRAME_D1, 0, 1000)
        prices[ticker] = pd.DataFrame(r)['close']
    df = pd.DataFrame(prices)

    print("\n" + "="*80)
    print(f"🔎 DIAGNOSTICA DI SELEZIONE - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("="*80)

    # Calcolo portafoglio
    selected_assets = build_selection_notebook(df, mem)
    
    print(f"\n✅ PORTAFOGLIO FINALE: {', '.join(selected_assets) if selected_assets else 'CASH'}")

    # Calcolo ordini
    final_orders = calcola_lotti_finali(selected_assets, df)

    print("\n" + "="*80)
    print("📋 TABELLA ORDINI ESECUTIVI")
    print("="*80)
    if final_orders:
        for ticker, size in final_orders.items():
            u = "Lotti" if ticker == 'GC=F' else "Azioni"
            print(f"👉 {ticker:<8} | Size: {size:>8} {u}")
    else:
        print("OPERATIVITÀ: TUTTO IN CASH")

    # Salvataggio stato
    if input("\nConfermare l'esecuzione e salvare lo stato? (s/n): ").lower() == 's':
        mem['active_positions_list'] = selected_assets
        # Aggiornamento prezzi di carico per Stop Loss
        new_prices = {}
        for asset in selected_assets:
            # Mantieni il vecchio prezzo se eri già dentro, altrimenti usa l'attuale
            new_prices[asset] = mem.get('active_prices', {}).get(asset, df[asset].iloc[-1])
        mem['active_prices'] = new_prices
        
        with open(STATE_FILE, 'w') as f:
            json.dump(mem, f, indent=4)
        print("💾 Stato salvato con successo.")

    mt5.shutdown()

if __name__ == "__main__":
    run_assistant()