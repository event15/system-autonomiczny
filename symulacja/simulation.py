#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from receptor import Receptor
from korelator import Korelator
from homeostat import Homeostat
from efektor import Efektor
import subprocess

MAKSYMALNY_CZAS_SYMULACJI = 80.0
KROK_CZASOWY_DT = 0.1

OPOZNIENIE_REJESTRACJI_TAU = 0.5
OPOZNIENIE_REAKCJI_TAU = 2.0
WAGA_ESTYMACJI_WT = 0.8
AMPLITUDA_SZUMU_RECEPTORA = 0.02
PRZEWODNOSC_POCZATKOWA_G0 = 0.1
ALFA_G_PRZEWODNOSCI = 0.05
EKSTYNKCJA_REJESTRACJI_R = 0.02
EKSTYNKCJA_DEREJESTRACJI_D = 0.01
MAKSYMALNA_MOC_EFEKTORA = 1.0
WSPOLCZYNNIK_VP_K = 0.05
WSPOLCZYNNIK_VP_DELTA_G = 2.0
K_VH_VP = 0.3
PROG_DECYZYJNY_VD = 0.1
WSPOLCZYNNIK_ROWNOWAGI_WR=0.05


def stworz_bodzce(wektor_czasu):
    lista_bodzcow = [
        {"start": 1.0, "stop": 2.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 2.5, "stop": 3.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 4.0, "stop": 5.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 5.5, "stop": 6.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 7.0, "stop": 8.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 8.5, "stop": 9.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 10.0, "stop": 11.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        # Nowe bodźce
        {"start": 11.5, "stop": 12.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 13.0, "stop": 14.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 14.5, "stop": 15.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 16.0, "stop": 17.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 17.5, "stop": 18.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 19.0, "stop": 20.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 20.5, "stop": 21.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 22.0, "stop": 23.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 23.5, "stop": 24.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 25.0, "stop": 26.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 26.5, "stop": 27.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 28.0, "stop": 29.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 29.5, "stop": 30.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 31.0, "stop": 32.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 32.5, "stop": 33.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 34.0, "stop": 35.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 35.5, "stop": 36.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 37.0, "stop": 38.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 38.5, "stop": 39.5, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
        {"start": 40.0, "stop": 41.0, "offset": 1.8, "amplitude_sin": 0.3, "frequency_sin": 0.0},
    ]
    suma_bodzcow = np.zeros_like(wektor_czasu)
    for bodziec in lista_bodzcow:
        indeks_startowy = int(bodziec["start"] / KROK_CZASOWY_DT)
        indeks_stopowy = int(bodziec["stop"] / KROK_CZASOWY_DT)
        indeks_startowy = max(0, min(indeks_startowy, len(wektor_czasu)))
        indeks_stopowy = max(0, min(indeks_stopowy, len(wektor_czasu)))
        segment_czasowy = wektor_czasu[indeks_startowy:indeks_stopowy]
        suma_bodzcow[indeks_startowy:indeks_stopowy] += (
                bodziec["offset"] + bodziec["amplitude_sin"] *
                np.sin(bodziec["frequency_sin"] * segment_czasowy)
        )
    return suma_bodzcow


def uruchom_symulacje():
    wektor_czasu = np.arange(0, MAKSYMALNY_CZAS_SYMULACJI, KROK_CZASOWY_DT)
    suma_bodzcow = stworz_bodzce(wektor_czasu)
    liczba_krokow_czasowych = len(wektor_czasu)

    receptor = Receptor(amplituda_szumu=AMPLITUDA_SZUMU_RECEPTORA, opoznienie_rejestracji_tau_s=OPOZNIENIE_REJESTRACJI_TAU)
    korelator = Korelator(przewodnosc_poczatkowa_g0=PRZEWODNOSC_POCZATKOWA_G0, alfa_g=ALFA_G_PRZEWODNOSCI,
                              ekstynkcja_r=EKSTYNKCJA_REJESTRACJI_R, ekstynkcja_d=EKSTYNKCJA_DEREJESTRACJI_D,
                              wspolczynnik_vp_k=WSPOLCZYNNIK_VP_K, wspolczynnik_vp_delta_g=WSPOLCZYNNIK_VP_DELTA_G)
    homeostat = Homeostat(wspolczynnik_rownowagi_wr=WSPOLCZYNNIK_ROWNOWAGI_WR,
                            prog_decyzyjny_vd=PROG_DECYZYJNY_VD, k_vh_vp=K_VH_VP)
    efektor = Efektor(Vd=PROG_DECYZYJNY_VD, opoznienie_reakcji_tau=OPOZNIENIE_REAKCJI_TAU,
                      P_max=MAKSYMALNA_MOC_EFEKTORA)

    wartosci_vr = np.zeros(liczba_krokow_czasowych)
    wartosci_vh = np.zeros(liczba_krokow_czasowych)
    wartosci_ve = np.zeros(liczba_krokow_czasowych)
    wartosci_g = np.zeros(liczba_krokow_czasowych)
    wartosci_k = np.zeros(liczba_krokow_czasowych)
    wartosci_vk = np.zeros(liczba_krokow_czasowych)
    wartosci_reakcji = np.zeros(liczba_krokow_czasowych)
    wartosci_vp = np.zeros(liczba_krokow_czasowych)
    wartosci_swiadomosci = np.zeros(liczba_krokow_czasowych)
    wartosci_myslenia = np.zeros(liczba_krokow_czasowych)
    wartosci_kg_graniczne = np.zeros(liczba_krokow_czasowych)

    czasy_decyzji = []
    czasy_reakcji = []
    zaplanowane_reakcje = []


    for i in range(liczba_krokow_czasowych):
        aktualny_czas = wektor_czasu[i]

        vr = receptor.procesuj_bodziec(suma_bodzcow[i], aktualny_czas)
        wartosci_vr[i] = vr

        vr_opozniony_input = receptor.pobierz_potencjal()

        korelator.oblicz_potencjal_korelacyjny(vr_opozniony_input, homeostat.pobierz_potencjal_refleksyjny())
        korelator.oblicz_moc_korelacyjna()
        korelator.oblicz_potencjal_estymacyjny(waga_wt=WAGA_ESTYMACJI_WT)

        korelator.aktualizuj_homeostat_vp(homeostat)
        homeostat.ustaw_potencjal_refleksyjny_proporcjonalnie()

        wartosci_vk[i] = korelator.pobierz_potencjal_korelacyjny_vk()
        wartosci_k[i] = korelator.pobierz_moc_korelacyjna()
        wartosci_ve[i] = korelator.pobierz_potencjal_estymacyjny()
        wartosci_vh[i] = homeostat.pobierz_potencjal_refleksyjny()
        wartosci_vp[i] = homeostat.pobierz_potencjal_perturbacyjny()
        wartosci_g[i] = korelator.pobierz_przewodnosc()
        wartosci_kg_graniczne[i] = korelator.pobierz_kg_graniczne() # Rejestruj Kg

        if suma_bodzcow[i] > 0:
            korelator.aktualizuj_przewodnosc_rejestracja(suma_bodzcow[i], KROK_CZASOWY_DT)
        else:
            korelator.aktualizuj_przewodnosc_derejestracja(KROK_CZASOWY_DT)

        sygnal_decyzji = efektor.procesuj_decyzje(wartosci_ve[i], aktualny_czas, ostatnia_reakcja=wartosci_reakcji[i-1] if i > 0 else 0)
        if sygnal_decyzji:
            czasy_decyzji.append(aktualny_czas)
            czas_zaplanowanej_reakcji = aktualny_czas + OPOZNIENIE_REAKCJI_TAU
            zaplanowane_reakcje.append({'czas': czas_zaplanowanej_reakcji, 'indeks': int(czas_zaplanowanej_reakcji / KROK_CZASOWY_DT)})

        efektor.wykonaj_reakcje(aktualny_czas, wartosci_ve[i], zaplanowane_reakcje, wartosci_reakcji, Vd=PROG_DECYZYJNY_VD, czasy_reakcji=czasy_reakcji, wektor_czasu=wektor_czasu)

        wartosci_swiadomosci[i] = wartosci_k[i] * (wartosci_vp[i] - wartosci_vh[i])
        wartosci_myslenia[i] = wartosci_myslenia[
                                 i - 1] + wartosci_k[i] * wartosci_vh[i] * KROK_CZASOWY_DT if i > 0 else 0

    return (wektor_czasu, suma_bodzcow, wartosci_vr, wartosci_vh, wartosci_ve, wartosci_g, wartosci_k, wartosci_vk,
            wartosci_reakcji, czasy_decyzji, czasy_reakcji, wartosci_vp, wartosci_swiadomosci, wartosci_myslenia, wartosci_kg_graniczne)


def generuj_wykresy_wynikow_symulacji(wektor_czasu, suma_bodzcow, wartosci_vr, wartosci_vh, wartosci_ve, wartosci_g, wartosci_k,
                                    wartosci_vk,
                                    wartosci_reakcji, czasy_decyzji, czasy_reakcji, wartosci_vp, wartosci_swiadomosci,
                                    wartosci_myslenia, wartosci_kg_graniczne):

    fig = plt.figure(figsize=(18, 21))
    kolory = {
        'bodziec': '#8A2BE2',
        'vr': '#4169E1',
        'vh': '#228B22',
        'Ve': '#8B4513',
        'G': '#FF8C00',
        'K': '#DC143C',
        'kg': '#9400D3',
        'vk': '#4B0082',
        'reakcja': '#B22222',
        'vp': '#6A5ACD',
        'swiadomosc': '#FF4500',
        'myslenie': '#4B0082'
    }
    xticks = np.arange(0, MAKSYMALNY_CZAS_SYMULACJI + 1, 10)
    etykieta_x = 'Czas [s]'

    # Subplot A: Potentials - Vr, Vh, Ve, Vk
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(wektor_czasu, suma_bodzcow, label='Bodziec (Suma)', color=kolory['bodziec'], lw=1.0, alpha=0.7)
    ax1.plot(wektor_czasu, wartosci_vr, label='Vr', color=kolory['vr'], alpha=0.8, linestyle='--')
    ax1.plot(wektor_czasu, wartosci_vh, label='Vh', color=kolory['vh'], alpha=0.8, linestyle=':')
    ax1.plot(wektor_czasu, wartosci_vk, label='Vk', color=kolory['vk'], lw=1.0)
    ax1.plot(wektor_czasu, wartosci_ve, label='Ve', color=kolory['Ve'], lw=1.0, linestyle='-.')
    ax1.axhline(PROG_DECYZYJNY_VD, color='black', linestyle='-.', label=f'Próg Vd')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.8)
    ax1.set_ylabel('Potencjał', fontsize=8)
    ax1.set_title('A) Potencjały: Vr, Vh, Vk, Ve i Bodziec', fontsize=9, pad=6)

    # Subplot B: Przewodność G(t), Moc K(t) i Kg(t)
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(wektor_czasu, wartosci_g, label='Przewodność G(t)', color=kolory['G'], lw=1.0)
    ax2.plot(wektor_czasu, wartosci_k, label='Moc K(t)', color=kolory['K'], alpha=0.8, linestyle='--')
    ax2.plot(wektor_czasu, wartosci_kg_graniczne, label='Kg(t) Graniczna Moc K', color=kolory['kg'], alpha=0.8, linestyle=':') # Added Kg
    ax2.fill_between(wektor_czasu, 0, wartosci_k, where=(wartosci_k > 0), color='#FFB6C1', alpha=0.2)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([])
    ax2.legend(loc='upper right', fontsize=7)
    ax2.set_ylabel('Wartość', fontsize=8)
    ax2.set_title('B) Przewodność G(t), Moc K(t) i Graniczna Moc Kg(t)', fontsize=9, pad=6) # Updated title

    # Subplot C: Feedback Loop - Vp, Vh, K
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(wektor_czasu, wartosci_vp, label='Vp (Perturbacja)', color=kolory['vp'], lw=1.0)
    ax3.plot(wektor_czasu, wartosci_vh, label='Vh (Refleksja)', color=kolory['vh'], alpha=0.8, linestyle='--')
    ax3.plot(wektor_czasu, wartosci_k, label='Moc K(t)', color=kolory['K'], alpha=0.8, linestyle=':') # Added K to feedback plot
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='-.') # Added horizontal line at 0
    ax3.fill_between(wektor_czasu, wartosci_vh, wartosci_vp, where=(wartosci_vp > wartosci_vh), facecolor='#90EE90', alpha=0.3, label='Przepływ Pozytywny (+)')
    ax3.fill_between(wektor_czasu, wartosci_vh, wartosci_vp, where=(wartosci_vp <= wartosci_vh), facecolor='#FFB6C1', alpha=0.3, label='Przepływ Negatywny (-)')
    ax3.set_xticks(xticks)
    ax3.set_xticklabels([])
    ax3.legend(loc='upper left', fontsize=7)
    ax3.set_ylabel('Potencjał / Moc', fontsize=8) # Updated y label
    ax3.set_title('C) Sprzężenie Zwrotne: Vp, Vh i Moc K(t)', fontsize=9, pad=6) # Updated title

    # Subplot D: Reakcja Efektora and Estymacja Ve(t)
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.step(wektor_czasu, wartosci_reakcji, label='Reakcja Efektora', color=kolory['reakcja'], where='post')
    ax4.plot(wektor_czasu, wartosci_ve, label='Ve', color=kolory['Ve'], alpha=0.8, linestyle='--')
    ax4.axhline(PROG_DECYZYJNY_VD, color='black', linestyle='-.', label=f'Próg Vd')
    ax4.set_ylim(-0.1, 1.5)
    ax4.set_yticks([0, 1])
    ax4.set_xticks(xticks)
    ax4.set_xticklabels([])
    ax4.legend(loc='upper right', fontsize=7)
    ax4.set_ylabel('Wartość', fontsize=8)
    ax4.set_title(f'D) Reakcja Efektora i Estymacja Ve(t) (Opóźnienie ∆td={OPOZNIENIE_REAKCJI_TAU}s)', fontsize=9, pad=6)

    # Subplot E: Decyzje and Reakcje Times
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.vlines(czasy_decyzji, ymin=0, ymax=1, colors='green', label='Czas Decyzji', lw=1.0)
    ax5.vlines(czasy_reakcji, ymin=0, ymax=1, colors='red', label='Czas Reakcji', lw=1.0)
    ax5.set_xticks(xticks)
    ax5.set_xticklabels( [f'{xtick:.0f}' for xtick in xticks])
    ax5.set_xlabel(etykieta_x, fontsize=8)
    ax5.set_yticks([])
    ax5.legend(loc='upper right', fontsize=7)
    ax5.set_title('E) Czasy Decyzji i Reakcji', fontsize=9, pad=6)

    # Subplot F: Świadomość and Myślenie
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(wektor_czasu, wartosci_swiadomosci, label='Świadomość (K·ΔV)', color=kolory['swiadomosc'], lw=1.0)
    ax6.plot(wektor_czasu, wartosci_myslenia, label='Myślenie (∫K·Vh dt)', color=kolory['myslenie'], linestyle='--', alpha=0.8)
    ax6.set_xticks(xticks)
    ax6.set_xticklabels( [f'{xtick:.0f}' for xtick in xticks])
    ax6.set_xlabel(etykieta_x, fontsize=8)
    ax6.set_ylabel('Wartość', fontsize=8)
    ax6.legend(loc='upper left', fontsize=7)
    ax6.set_title('F) Świadomość i Myślenie', fontsize=9, pad=6)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()


def main():
    """Główna funkcja do uruchomienia symulacji i wygenerowania wykresów."""
    try:
        wyniki_symulacji = uruchom_symulacje()
        generuj_wykresy_wynikow_symulacji(*wyniki_symulacji)

        wektor_czasu, suma_bodzcow, wartosci_vr, wartosci_vh, wartosci_ve, wartosci_g, wartosci_k, wartosci_vk, wartosci_reakcji, czasy_decyzji, czasy_reakcji, wartosci_vp, wartosci_swiadomosci, wartosci_myslenia, wartosci_kg_graniczne = wyniki_symulacji

        df_wyniki = pd.DataFrame({
            't': wektor_czasu,
            'bodziec': suma_bodzcow,
            'Vr': wartosci_vr,
            'Vh': wartosci_vh,
            'Ve': wartosci_ve,
            'G': wartosci_g,
            'K': wartosci_k,
            'Vk': wartosci_vk,
            'reakcja': wartosci_reakcji,
            'Vp': wartosci_vp,
            'swiadomosc': wartosci_swiadomosci,
            'myslenie': wartosci_myslenia,
            'Kg_graniczne': wartosci_kg_graniczne
        })

        tablica_decyzji = np.zeros_like(wektor_czasu)
        for czas_decyzji in czasy_decyzji:
            indeks_decyzji = np.argmin(np.abs(wektor_czasu - czas_decyzji))
            tablica_decyzji[indeks_decyzji] = 1
        df_wyniki['decyzja'] = tablica_decyzji

        nazwa_pliku_csv = "wyniki_symulacji.csv"
        df_wyniki.to_csv(nazwa_pliku_csv, index=False)
        print(f"Wyniki symulacji zapisane do {nazwa_pliku_csv}")

        # subprocess.run(['python', 'symulacja/analiza.py'])

    except KeyboardInterrupt:
        print("\nProgram przerwany przez użytkownika (Ctrl+C)")
        sys.exit(0)

if __name__ == "__main__":
    main()
