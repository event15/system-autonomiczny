import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    PINGOUIN_AVAILABLE = True
    from pingouin import partial_corr
except ImportError:
    PINGOUIN_AVAILABLE = False
    partial_corr = None

try:
    from dtw import dtw as dtw_algo
    from dtw.stepPattern import rabinerJuangStepPattern
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


def time_lagged_corr(x, y, max_lag=10):
    """Korelacja z przesunięciem (time-lagged)."""
    wyniki = {}
    for lag in range(-max_lag, max_lag+1):
        if lag > 0:
            xx = x[:-lag]
            yy = y[lag:]
        elif lag < 0:
            xx = x[-lag:]
            yy = y[:len(y)+lag]
        else:
            xx = x
            yy = y
        if len(xx)>1 and len(yy)>1:
            r, _ = pearsonr(xx, yy)
            wyniki[lag] = r
        else:
            wyniki[lag] = np.nan
    return wyniki

def analyze_windows_around_decisions(df, window_pre=1.0, window_post=1.0):
    """
    Dla każdej decyzji, wycinamy okno +/- i liczymy min/max/mean z kolumn.
    """
    results = []
    t = df['t'].values
    dt_ = (t[1]-t[0]) if len(t)>1 else 0.1
    n_step_pre = int(window_pre / dt_)
    n_step_post= int(window_post / dt_)

    skip_cols = ('t','czas','decyzja','reakcja','sw_segment','time_to_next_dec')
    cols = [c for c in df.columns if c not in skip_cols]

    dec_indexes = df.index[df['decyzja']==1].tolist()
    for idx in dec_indexes:
        dec_time = df.loc[idx,'t']
        low  = max(0, idx - n_step_pre)
        high = min(len(df), idx + n_step_post + 1)
        window = df.loc[low:high, cols]
        stats_min  = window.min()
        stats_max  = window.max()
        stats_mean = window.mean()

        row = {
            'dec_time': dec_time,
            'idx_dec' : idx,
            'win_pre' : window_pre,
            'win_post': window_post
        }
        for c_ in cols:
            row[f"{c_}_min"]  = stats_min[c_]
            row[f"{c_}_max"]  = stats_max[c_]
            row[f"{c_}_mean"] = stats_mean[c_]
        results.append(row)

    return pd.DataFrame(results)

def define_epizodes_for_all(df, threshold_factor=0.3):
    """
    Szuka przedziałów w których zmienna > threshold_factor * max_zmiennej.
    Dla wszystkich kolumn (poza binarnymi).
    """
    results = {}
    skip_cols = ['t','czas','decyzja','reakcja','time_to_next_dec','sw_segment']
    numeric_cols = [c for c in df.columns if c not in skip_cols]
    for col in numeric_cols:
        max_val = df[col].max()
        thr = threshold_factor * max_val
        seg = np.where(df[col]>thr,1,0)

        epizody = []
        in_ep = False
        start_i = None
        for i,v in enumerate(seg):
            if v==1 and not in_ep:
                in_ep = True
                start_i = i
            elif v==0 and in_ep:
                in_ep = False
                end_i = i-1
                epizody.append((start_i,end_i))
        if in_ep:
            epizody.append((start_i, len(seg)-1))

        results[col] = []
        for (s,e) in epizody:
            t_start = df['t'].iloc[s]
            t_end   = df['t'].iloc[e]
            local_max = df[col].iloc[s:e+1].max()
            results[col].append((s,e,t_start,t_end,local_max))
    return results

def correlation_with_time_to_decision(df, colname='Vh'):
    """
    Dla każdej chwili obliczamy time_to_next_dec.
    Potem liczymy korelację (Pearson) z zadaną kolumną.
    """
    t = df['t'].values
    decision_times = t[df['decyzja']==1]
    time_to_next = []
    for i in range(len(df)):
        current_t = t[i]
        future = decision_times[decision_times>=current_t]
        if len(future)>0:
            dist = future[0] - current_t
        else:
            dist = np.nan
        time_to_next.append(dist)
    df['time_to_next_dec'] = time_to_next

    valid = df.dropna(subset=['time_to_next_dec', colname])
    if len(valid)>5:
        r, p = pearsonr(valid[colname], valid['time_to_next_dec'])
        return r,p, len(valid)
    else:
        return np.nan, np.nan, len(valid)

def visualize_histograms(df, window_pre=1.0):
    """
    Porównanie histogramów wartości (np. swiadomosc, myslenie, Vh, Ve)
    w oknie 1s przed decyzją i 1s przed "non-decision".
    """
    import random
    t = df['t'].values
    dt_ = (t[1]-t[0]) if len(t)>1 else 0.1
    n_step_pre = int(window_pre / dt_)

    var_list = ['swiadomosc','myslenie','Vh','Ve']

    pre_dec_values = {v:[] for v in var_list}
    no_dec_values  = {v:[] for v in var_list}

    dec_idxs = df.index[df['decyzja']==1].tolist()
    # 1) przed decyzją
    for idx in dec_idxs:
        low = max(0, idx-n_step_pre)
        chunk = df.loc[low:idx,var_list]
        for v in var_list:
            pre_dec_values[v].extend(chunk[v].values)

    # 2) analogicznie "przed brakiem decyzji" (losowo)
    no_dec_idxs = df.index[df['decyzja']==0].tolist()
    sample_size = min(len(no_dec_idxs), len(dec_idxs))
    random_idxs = random.sample(no_dec_idxs, sample_size)
    for rid in random_idxs:
        low = max(0, rid-n_step_pre)
        chunk = df.loc[low:rid, var_list]
        for v in var_list:
            no_dec_values[v].extend(chunk[v].values)

    fig, axes = plt.subplots(len(var_list), 1, figsize=(10,4*len(var_list)))
    for i,v in enumerate(var_list):
        ax = axes[i]
        ax.hist(pre_dec_values[v], bins=30, alpha=0.5, label=f'{v} przed decyzją')
        ax.hist(no_dec_values[v], bins=30, alpha=0.5, label=f'{v} przed non-decyzją')
        ax.set_title(f"Histogram: {v}, okno {window_pre}s przed (decyzja vs. brak)")
        ax.legend()
    plt.tight_layout()
    plt.show()

def visualize_correlation_heatmap(df, cols=None):
    """
    Heatmapa korelacji (Pearson) dla wskazanych kolumn.
    """
    if cols is None:
        cols = ['Vr','Vh','Ve','K','swiadomosc','myslenie','decyzja','reakcja']
    c = df[cols].corr()
    plt.figure(figsize=(8,6))
    plt.imshow(c, cmap='bwr', vmin=-1, vmax=1)
    plt.xticks(range(len(cols)), cols, rotation=45)
    plt.yticks(range(len(cols)), cols)
    plt.colorbar(label="Pearson R")
    plt.title("Heatmap korelacji")
    plt.show()


def granger_causality_analysis(df, max_lag=5):
    results = {}
    # Poprawione przygotowanie danych - kolumna 1: Y, kolumna 0: X
    data_1 = df[['myslenie', 'swiadomosc']].dropna().values
    res_1 = grangercausalitytests(data_1, max_lag, addconst=True, verbose=False)

    # Ekstrakcja właściwych wyników testów
    clean_results = {}
    for lag, value in res_1.items():
        test_results = value[0]  # Pierwszy element krotki to wyniki testów
        clean_results[lag] = test_results
    results['myslenie_~_swiadomosc'] = clean_results

    # Analogicznie dla drugiego kierunku
    data_2 = df[['swiadomosc', 'myslenie']].dropna().values
    res_2 = grangercausalitytests(data_2, max_lag, addconst=True, verbose=False)

    clean_results_2 = {}
    for lag, value in res_2.items():
        test_results = value[0]
        clean_results_2[lag] = test_results
    results['swiadomosc_~_myslenie'] = clean_results_2

    return results

def interpret_granger_results(granger_dict):
    """
    Proste wypisanie w pętli: F-stat i p-value dla kolejnych lagów.
    """
    for label, lag_dict in granger_dict.items():
        print(f"\n=== Granger test: {label} ===")
        for lag, metrics in lag_dict.items():
            # Sprawdź, czy 'ssr_ftest' istnieje i ma oczekiwaną strukturę
            if 'ssr_ftest' in metrics:
                ssr_ftest = metrics['ssr_ftest']
                # Sprawdź, czy to krotka (stara wersja) lub słownik (nowsza wersja)
                if isinstance(ssr_ftest, tuple) and len(ssr_ftest) >= 2:
                    f_stat, p_val = ssr_ftest[0], ssr_ftest[1]
                else:
                    # Jeśli struktura jest inna, ustaw domyślne wartości
                    f_stat, p_val = np.nan, np.nan
                    print(f"Nieoczekiwana struktura 'ssr_ftest' dla lag {lag} w {label}")
            else:
                f_stat, p_val = np.nan, np.nan
                print(f"Brak 'ssr_ftest' dla lag {lag} w {label}")
            print(f"Lag={lag}: F={f_stat:.3f}, p={p_val:.3g}")


def dynamic_time_warping_analysis(df):
    """
    DTW: porównanie przebiegów 'swiadomosc' vs. 'myslenie'.
    Wymaga biblioteki dtw (pip install dtw-python).
    """
    if not DTW_AVAILABLE:
        print("DTW library not installed. (pip install dtw-python)")
        return

    sigA = df['swiadomosc'].dropna().values
    sigB = df['myslenie'].dropna().values
    step_pattern = rabinerJuangStepPattern(6, "c")

    # Poprawiona nazwa funkcji dtw_algo (zgodna z importem)
    alignment = dtw_algo(
        sigA, sigB,
        keep_internals=True,
        step_pattern=step_pattern
    )

    dist = alignment.distance
    print(f"DTW distance(swiadomosc, myslenie) = {dist:.4f}")

def animate_correlation_heatmap(df, cols, window_size=50, step=10, interval=300):
    """
    Tworzy animowaną heatmapę korelacji w ruchomych oknach (dla kolumn 'cols').
    """
    fig, ax = plt.subplots(figsize=(6,5))

    corr_data = df[cols].iloc[0:window_size].corr()
    im = ax.imshow(corr_data, cmap='bwr', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)

    cbar = fig.colorbar(im)
    cbar.set_label('Pearson R')
    title_text = ax.set_title("Correlation Heatmap (Frame 0)")

    starts = np.arange(0, len(df)-window_size, step)

    def update(frame_idx):
        start_idx = starts[frame_idx]
        end_idx   = start_idx + window_size
        sub = df[cols].iloc[start_idx:end_idx]
        c = sub.corr()
        im.set_data(c.values)
        title_text.set_text(f"Samples {start_idx}..{end_idx}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(starts), interval=interval, blit=False)
    plt.close(fig)
    return anim


def run_analyses(df, scenario_name="scenario", do_advanced=False):
    """
    Uruchamia analizy statystyczne na DataFrame.
    """

    # --- Podstawowe analizy:
    print(f"\n--- {scenario_name}: Pairwise correlation (bez lag) ---")
    print(df.corr())

    visualize_correlation_heatmap(df)

    # Time-lag example
    print(f"\n--- {scenario_name}: Time-lagged cross-corr (swiadomosc vs myslenie) ---")
    tlc = time_lagged_corr(df['swiadomosc'].values, df['myslenie'].values, max_lag=10)
    for lg,val in sorted(tlc.items()):
        print(f"lag={lg}: r={val:.4f}")

    # Regresja (sklearn)
    print(f"\n--- {scenario_name}: Regresja logistyczna (sklearn) ---")
    X = df[['swiadomosc','myslenie','Vh','Ve']].values
    y = df['decyzja'].values
    if len(np.unique(y))>1:
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X,y)
        yhat = clf.predict(X)
        print("Wsp. regresji:", clf.coef_, "Intercept:", clf.intercept_)
        print(classification_report(y, yhat, digits=3, zero_division=0))
    else:
        print("Brak zróżnicowania w 'decyzja' (same 0 albo same 1).")

    # Regresja (statsmodels)
    print(f"\n--- {scenario_name}: Regresja logistyczna (statsmodels) ---")
    if len(np.unique(y))>1:
        X_sm = sm.add_constant(X, prepend=True)
        model_sm = sm.Logit(y, X_sm)
        try:
            res = model_sm.fit(disp=False)
            print(res.summary())
        except Exception as e:
            print("Logit błąd:", e)
    else:
        print("Brak zróżnicowania w 'decyzja'.")

    # Analiza warunkowa (min/max w oknach)
    stats_dec = analyze_windows_around_decisions(df,1.0,1.0)
    stats_dec.to_csv(f"{scenario_name}_analysis_windows.csv", index=False)
    print(f"\n--- {scenario_name}: Okna wokół decyzji ---")
    print(stats_dec.head())

    # Epizody
    print(f"\n--- {scenario_name}: Epizody > 30% max (wszystkie zmienne) ---")
    ep_all = define_epizodes_for_all(df, threshold_factor=0.3)
    for col, eplst in ep_all.items():
        print(f"{col}: {len(eplst)} epizodów")
        for (s,e,ts,te,mxv) in eplst:
            print(f"   t={ts:.2f}..{te:.2f}, max={mxv:.3f}")

    # Korelacja z time_to_next_dec
    print(f"\n--- {scenario_name}: correlation z time_to_next_dec ---")
    for var_ in ['Vr','Vh','Ve','Vp','swiadomosc','myslenie']:
        r,p,nv = correlation_with_time_to_decision(df, var_)
        print(f"{var_}: r={r:.3f}, p={p:.3e}, n={nv}")

    # Wizualizacja histogramów 1s przed decyzją
    visualize_histograms(df, window_pre=1.0)

    # Korelacja częściowa
    if PINGOUIN_AVAILABLE:
        print(f"\n--- {scenario_name}: partial_corr(swiadomosc, Ve | K) ---")
        df_tmp = df[['swiadomosc','Ve','K']].dropna()
        if len(df_tmp)>5:
            pcres = partial_corr(data=df_tmp, x='swiadomosc', y='Ve', covar='K', method='pearson')
            print(pcres)
        else:
            print("Za mało danych do partial_corr.")
    else:
        print("Pingouin nie zainstalowane - pomijamy partial_corr")

    # --- Analizy zaawansowane, jeśli "do_advanced=True"
    if do_advanced:
        print(f"\n+++ {scenario_name}: Granger Causality +++")
        gdict = granger_causality_analysis(df, max_lag=3)
        interpret_granger_results(gdict)

        print(f"\n+++ {scenario_name}: Dynamic Time Warping +++")
        dynamic_time_warping_analysis(df)

        print(f"\n+++ {scenario_name}: Animowana Heatmapa (kolumny: swiadomosc, myslenie, Ve, Vh) +++")
        anim = animate_correlation_heatmap(df, cols=['swiadomosc','myslenie','Ve','Vh'],
                                           window_size=50, step=10, interval=300)
        # Możesz zapisać anim:
        plt.show()
        anim.save(f"{scenario_name}_heatmap.gif", writer="pillow")
        print("Animacja stworzona (zwrócony obiekt FuncAnimation)")

    print(f"\n=== Zakończono analizy dla {scenario_name} ===")


def main():
    """
    Główna funkcja uruchamiająca analizy.
    """
    csv_filename = "simulation_results.csv" # Nazwa pliku CSV z wynikami symulacji
    try:
        df_results = pd.read_csv(csv_filename)
        print(f"Data loaded from {csv_filename}, shape: {df_results.shape}")

        scenario_name = "simulation" # Nazwa scenariusza dla analiz (możesz dostosować)
        do_advanced_analysis = True  # Czy uruchamiać analizy zaawansowane (Granger, DTW, animacja)

        run_analyses(df_results, scenario_name=scenario_name, do_advanced=do_advanced_analysis)

    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found. Make sure to run simulation.py first.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")


if __name__ == "__main__":
    main()