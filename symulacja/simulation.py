#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from receptor import Receptor
from korelator import Korelator
from homeostat import Homeostat
from efektor import Efektor
from matplotlib.widgets import Slider, Button

DEFAULT_MAKSYMALNY_CZAS_SYMULACJI = 80.0
DEFAULT_KROK_CZASOWY_DT = 0.1

DEFAULT_OPOZNIENIE_REJESTRACJI_TAU = 0.5
DEFAULT_OPOZNIENIE_REAKCJI_TAU = 2.0
DEFAULT_WAGA_ESTYMACJI_WT = 0.8
DEFAULT_AMPLITUDA_SZUMU_RECEPTORA = 0.02
DEFAULT_PRZEWODNOSC_POCZATKOWA_G0 = 0.1
DEFAULT_ALFA_G_PRZEWODNOSCI = 0.05
DEFAULT_EKSTYNKCJA_REJESTRACJI_R = 0.02
DEFAULT_EKSTYNKCJA_DEREJESTRACJI_D = 0.01
DEFAULT_MAKSYMALNA_MOC_EFEKTORA = 1.0
DEFAULT_WSPOLCZYNNIK_VP_K = 0.05
DEFAULT_WSPOLCZYNNIK_VP_DELTA_G = 2.0
DEFAULT_K_VH_VP = 0.3
DEFAULT_PROG_DECYZYJNY_VD = 0.1
DEFAULT_WSPOLCZYNNIK_ROWNOWAGI_WR = 0.05


def stworz_bodzce(wektor_czasu, bodziec_params):
    lista_bodzcow = [bodziec_params]
    suma_bodzcow = np.zeros_like(wektor_czasu)
    for bodziec in lista_bodzcow:
        indeks_startowy = int(bodziec["start"] / DEFAULT_KROK_CZASOWY_DT)
        indeks_stopowy = int(bodziec["stop"] / DEFAULT_KROK_CZASOWY_DT)
        indeks_startowy = max(0, min(indeks_startowy, len(wektor_czasu)))
        indeks_stopowy = max(0, min(indeks_stopowy, len(wektor_czasu)))
        segment_czasowy = wektor_czasu[indeks_startowy:indeks_stopowy]
        suma_bodzcow[indeks_startowy:indeks_stopowy] += (
                bodziec["offset"] + bodziec["amplitude_sin"] *
                np.sin(bodziec["frequency_sin"] * segment_czasowy)
        )
    return suma_bodzcow


def uruchom_symulacje(maksymalny_czas_symulacji=DEFAULT_MAKSYMALNY_CZAS_SYMULACJI,
                      krok_czasowy_dt=DEFAULT_KROK_CZASOWY_DT,
                      opoznienie_rejestracji_tau=DEFAULT_OPOZNIENIE_REJESTRACJI_TAU,
                      opoznienie_reakcji_tau=DEFAULT_OPOZNIENIE_REAKCJI_TAU,
                      waga_estymacji_wt=DEFAULT_WAGA_ESTYMACJI_WT,
                      amplituda_szumu_receptora=DEFAULT_AMPLITUDA_SZUMU_RECEPTORA,
                      przewodnosc_poczatkowa_g0=DEFAULT_PRZEWODNOSC_POCZATKOWA_G0,
                      alfa_g_przewodnosci=DEFAULT_ALFA_G_PRZEWODNOSCI,
                      ekstynkcja_rejestracji_r=DEFAULT_EKSTYNKCJA_REJESTRACJI_R,
                      ekstynkcja_derejestracji_d=DEFAULT_EKSTYNKCJA_DEREJESTRACJI_D,
                      maksymalna_moc_efektora=DEFAULT_MAKSYMALNA_MOC_EFEKTORA,
                      wspolczynnik_vp_k=DEFAULT_WSPOLCZYNNIK_VP_K,
                      wspolczynnik_vp_delta_g=DEFAULT_WSPOLCZYNNIK_VP_DELTA_G,
                      k_vh_vp=DEFAULT_K_VH_VP,
                      prog_decyzyjny_vd=DEFAULT_PROG_DECYZYJNY_VD,
                      wspolczynnik_rownowagi_wr=DEFAULT_WSPOLCZYNNIK_ROWNOWAGI_WR):

    wektor_czasu = np.arange(0, maksymalny_czas_symulacji, krok_czasowy_dt)
    bodziec_params = {"start": 1.0, "stop": 12.0, "offset": 2.8, "amplitude_sin": 0.3, "frequency_sin": 0.0}
    suma_bodzcow = stworz_bodzce(wektor_czasu, bodziec_params)
    liczba_krokow_czasowych = len(wektor_czasu)

    receptor = Receptor(amplituda_szumu=amplituda_szumu_receptora, opoznienie_rejestracji_tau_s=opoznienie_rejestracji_tau)
    korelator = Korelator(przewodnosc_poczatkowa_g0=przewodnosc_poczatkowa_g0, alfa_g=alfa_g_przewodnosci * 5,
                              ekstynkcja_r=ekstynkcja_rejestracji_r * 5,
                              ekstynkcja_d=ekstynkcja_derejestracji_d * 5,
                              wspolczynnik_vp_k=wspolczynnik_vp_k, wspolczynnik_vp_delta_g=wspolczynnik_vp_delta_g)
    homeostat = Homeostat(wspolczynnik_rownowagi_wr=wspolczynnik_rownowagi_wr,
                            prog_decyzyjny_vd=prog_decyzyjny_vd, k_vh_vp=k_vh_vp)
    efektor = Efektor(Vd=prog_decyzyjny_vd, opoznienie_reakcji_tau=opoznienie_reakcji_tau,
                      P_max=maksymalna_moc_efektora, czas_trwania_reakcji_tau_r=7.0)

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
    wartosci_delta_g = np.zeros(liczba_krokow_czasowych)

    czasy_decyzji = []
    czasy_reakcji = []
    zaplanowane_reakcje = []
    vk_decyzje = []

    poprzednia_g = przewodnosc_poczatkowa_g0

    for i in range(liczba_krokow_czasowych):
        aktualny_czas = wektor_czasu[i]

        vr = receptor.procesuj_bodziec(suma_bodzcow[i], aktualny_czas, krok_czasowy_dt)
        wartosci_vr[i] = vr

        vr_opozniony_input = receptor.pobierz_potencjal()

        korelator.oblicz_potencjal_korelacyjny(vr_opozniony_input, homeostat.pobierz_potencjal_refleksyjny())
        korelator.oblicz_moc_korelacyjna()
        korelator.oblicz_potencjal_estymacyjny(waga_wt=waga_estymacji_wt)

        korelator.aktualizuj_homeostat_vp(homeostat)
        homeostat.ustaw_potencjal_refleksyjny_proporcjonalnie()

        wartosci_vk[i] = korelator.pobierz_potencjal_korelacyjny_vk()
        wartosci_k[i] = korelator.pobierz_moc_korelacyjna()
        wartosci_ve[i] = korelator.pobierz_potencjal_estymacyjny()
        wartosci_vh[i] = homeostat.pobierz_potencjal_refleksyjny()
        wartosci_vp[i] = homeostat.pobierz_potencjal_perturbacyjny()
        wartosci_g[i] = korelator.pobierz_przewodnosc()
        wartosci_kg_graniczne[i] = korelator.pobierz_kg_graniczne()
        wartosci_delta_g[i] = abs(wartosci_g[i] - poprzednia_g)
        poprzednia_g = wartosci_g[i]


        if suma_bodzcow[i] > 0:
            korelator.aktualizuj_przewodnosc_rejestracja(suma_bodzcow[i], krok_czasowy_dt)
        else:
            korelator.aktualizuj_przewodnosc_derejestracja(krok_czasowy_dt)

        sygnal_decyzji = efektor.procesuj_decyzje(wartosci_ve[i], aktualny_czas, ostatnia_reakcja=wartosci_reakcji[i-1] if i > 0 else 0)
        if sygnal_decyzji:
            czasy_decyzji.append(aktualny_czas)
            vk_decyzje.append(wartosci_vk[i])
            czas_zaplanowanej_reakcji = aktualny_czas + opoznienie_reakcji_tau
            zaplanowane_reakcje.append({'czas': czas_zaplanowanej_reakcji, 'indeks': int(czas_zaplanowanej_reakcji / krok_czasowy_dt)})

        efektor.wykonaj_reakcje(aktualny_czas, wartosci_ve[i], zaplanowane_reakcje, wartosci_reakcji, Vd=prog_decyzyjny_vd, czasy_reakcji=czasy_reakcji, wektor_czasu=wektor_czasu)

        wartosci_swiadomosci[i] = wartosci_k[i] * (wartosci_vp[i] - wartosci_vh[i])
        wartosci_myslenia[i] = wartosci_myslenia[
                                 i - 1] + wartosci_k[i] * wartosci_vh[i] * krok_czasowy_dt if i > 0 else 0

    return (wektor_czasu, suma_bodzcow, wartosci_vr, wartosci_vh, wartosci_ve, wartosci_g, wartosci_k, wartosci_vk,
            wartosci_reakcji, czasy_decyzji, czasy_reakcji, wartosci_vp, wartosci_swiadomosci, wartosci_myslenia, wartosci_kg_graniczne, wartosci_delta_g, vk_decyzje)

def generuj_wykresy_wynikow_symulacji(fig, osie, wektor_czasu, suma_bodzcow, wartosci_vr, wartosci_vh, wartosci_ve, wartosci_g, wartosci_k,
                                    wartosci_vk,
                                    wartosci_reakcji, czasy_decyzji, czasy_reakcji, wartosci_vp, wartosci_swiadomosci,
                                    wartosci_myslenia, wartosci_kg_graniczne, wartosci_delta_g, vk_decyzje, prog_decyzyjny_vd, opoznienie_reakcji_tau):

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
    xticks = np.arange(0, wektor_czasu[-1] + 1, 10) if len(wektor_czasu) > 0 else []
    etykieta_x = 'Czas [s]'

    # Subplot A: Potentials - Vr, Vh, Ve, Vk
    ax1 = osie[0, 0]
    ax1.clear()
    ax1.plot(wektor_czasu, suma_bodzcow, label='Bodziec (Suma)', color=kolory['bodziec'], lw=1.0, alpha=0.7)
    ax1.plot(wektor_czasu, wartosci_vr, label='Vr', color=kolory['vr'], alpha=0.8, linestyle='--')
    ax1.plot(wektor_czasu, wartosci_vh, label='Vh', color=kolory['vh'], alpha=0.8, linestyle=':')
    ax1.plot(wektor_czasu, wartosci_vk, label='Vk', color=kolory['vk'], lw=1.0)
    ax1.plot(wektor_czasu, wartosci_ve, label='Ve', color=kolory['Ve'], lw=1.0, linestyle='-.')
    ax1.axhline(prog_decyzyjny_vd, color='black', linestyle='-.', label=f'Próg Vd')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.8)
    ax1.set_ylabel('Potencjał', fontsize=8)
    ax1.set_title('A) Potencjały: Vr, Vh, Vk, Ve i Bodziec', fontsize=9, pad=6)

    # Subplot B: Przewodność G(t), Moc K(t) i Kg(t)
    ax2 = osie[0, 1]
    ax2.clear()
    ax2.plot(wektor_czasu, wartosci_g, label='Przewodność G(t)', color=kolory['G'], lw=1.0)
    ax2.plot(wektor_czasu, wartosci_k, label='Moc K(t)', color=kolory['K'], alpha=0.8, linestyle='--')
    ax2.plot(wektor_czasu, wartosci_kg_graniczne, label='Kg(t) Graniczna Moc K', color=kolory['kg'], alpha=0.8, linestyle=':')
    ax2.fill_between(wektor_czasu, 0, wartosci_k, where=(wartosci_k > 0), color='#FFB6C1', alpha=0.2)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([])
    ax2.legend(loc='upper right', fontsize=7)
    ax2.set_ylabel('Wartość', fontsize=8)
    ax2.set_title('B) Przewodność G(t), Moc K(t) i Graniczna Moc Kg(t)', fontsize=9, pad=6)

    # Subplot C: Feedback Loop - Vp, Vh, K
    ax3 = osie[0, 2]
    ax3.clear()
    ax3.plot(wektor_czasu, wartosci_vp, label='Vp (Perturbacja)', color=kolory['vp'], lw=1.0)
    ax3.plot(wektor_czasu, wartosci_vh, label='Vh (Refleksja)', color=kolory['vh'], alpha=0.8, linestyle='--')
    ax3.plot(wektor_czasu, wartosci_k, label='Moc K(t)', color=kolory['K'], alpha=0.8, linestyle=':')
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='-.')
    ax3.fill_between(wektor_czasu, wartosci_vh, wartosci_vp, where=(wartosci_vp > wartosci_vh), facecolor='#90EE90', alpha=0.3, label='Przepływ Pozytywny (+)')
    ax3.fill_between(wektor_czasu, wartosci_vh, wartosci_vp, where=(wartosci_vp <= wartosci_vh), facecolor='#FFB6C1', alpha=0.3, label='Przepływ Negatywny (-)')
    ax3.set_xticks(xticks)
    ax3.set_xticklabels([])
    ax3.legend(loc='upper left', fontsize=7)
    ax3.set_ylabel('Potencjał / Moc', fontsize=8)
    ax3.set_title('C) Sprzężenie Zwrotne: Vp, Vh i Moc K(t)', fontsize=9, pad=6)

    # Subplot D: Reakcja Efektora and Estymacja Ve(t)
    ax4 = osie[1, 0]
    ax4.clear()
    ax4.step(wektor_czasu, wartosci_reakcji, label='Reakcja Efektora', color=kolory['reakcja'], where='post')
    ax4.plot(wektor_czasu, wartosci_ve, label='Ve', color=kolory['Ve'], alpha=0.8, linestyle='--')
    ax4.axhline(prog_decyzyjny_vd, color='black', linestyle='-.', label=f'Próg Vd')
    ax4.set_ylim(-0.1, 1.5)
    ax4.set_yticks([0, 1])
    ax4.set_xticks(xticks)
    ax4.set_xticklabels([])
    ax4.legend(loc='upper right', fontsize=7)
    ax4.set_ylabel('Wartość', fontsize=8)
    ax4.set_title(f'D) Reakcja Efektora i Estymacja Ve(t) (Opóźnienie ∆td={opoznienie_reakcji_tau:.2f}s)', fontsize=9, pad=6)

    # Subplot E: Decyzje and Reakcje Times
    ax5 = osie[1, 1]
    ax5.clear()
    ax5.vlines(czasy_decyzji, ymin=0, ymax=1, colors='green', label='Czas Decyzji', lw=1.0)
    ax5.vlines(czasy_reakcji, ymin=0, ymax=1, colors='red', label='Czas Reakcji', lw=1.0)
    ax5.set_xticks(xticks)
    ax5.set_xticklabels( [f'{xtick:.0f}' for xtick in xticks])
    ax5.set_xlabel(etykieta_x, fontsize=8)
    ax5.set_yticks([])
    ax5.legend(loc='upper right', fontsize=7)
    ax5.set_title('E) Czasy Decyzji i Reakcji', fontsize=9, pad=6)

    # Subplot F: Świadomość and Myślenie
    ax6 = osie[1, 2]
    ax6.clear()
    ax6.plot(wektor_czasu, wartosci_swiadomosci, label='Świadomość (K·ΔV)', color=kolory['swiadomosc'], lw=1.0)
    ax6.plot(wektor_czasu, wartosci_myslenia, label='Myślenie (∫K·Vh dt)', color=kolory['myslenie'], linestyle='--', alpha=0.8)
    ax6.set_xticks(xticks)
    ax6.set_xticklabels( [f'{xtick:.0f}' for xtick in xticks])
    ax6.set_xlabel(etykieta_x, fontsize=8)
    ax6.set_ylabel('Wartość', fontsize=8)
    ax6.legend(loc='upper left', fontsize=7)
    ax6.set_title('F) Świadomość i Myślenie', fontsize=9, pad=6)

    # Subplot G: Przewodność G(t) - DETAL
    ax7 = osie[2, 0]
    ax7.clear()
    ax7.plot(wektor_czasu, wartosci_g, label='Przewodność G(t)', color=kolory['G'], lw=1.0)
    ax7.set_xticks(xticks)
    ax7.set_xticklabels([f'{xtick:.0f}' for xtick in xticks])
    ax7.set_xlabel(etykieta_x, fontsize=8)
    ax7.set_ylabel('Przewodność G', fontsize=8)
    ax7.legend(loc='upper right', fontsize=7)
    ax7.set_title('G) Przewodność G(t) - DETAL', fontsize=9, pad=6)

    # Subplot H: Przyrost Przewodności Delta G(t)
    ax8 = osie[2, 1]
    ax8.clear()
    ax8.plot(wektor_czasu, wartosci_delta_g, label='Przyrost ΔG(t)', color=kolory['G'], lw=1.0)
    ax8.set_xticks(xticks)
    ax8.set_xticklabels([f'{xtick:.0f}' for xtick in xticks])
    ax8.set_xlabel(etykieta_x, fontsize=8)
    ax8.set_ylabel('Przyrost ΔG', fontsize=8)
    ax8.legend(loc='upper right', fontsize=7)
    ax8.set_title('H) Przyrost Przewodności ΔG(t)', fontsize=9, pad=6)

    # Subplot I: Czas Decyzji vs. Potencjał Korelacyjny Vk_decyzja
    ax9 = osie[2, 2]
    ax9.clear()
    ax9.scatter(czasy_decyzji, vk_decyzje, color=kolory['vk'], alpha=0.8)
    ax9.set_ylabel('Potencjał Korelacyjny Vk w chwili decyzji', fontsize=8)
    ax9.set_xlabel('Czas Decyzji [s]', fontsize=8)
    ax9.set_title('I) Czas Decyzji vs. Potencjał Korelacyjny Vk', fontsize=9, pad=6)

    # Subplot J: Zależność Ve(Vk) z Asymptotą
    ax10 = osie[3, 0]
    ax10.clear()
    ax10.plot(wartosci_vk, wartosci_ve, color=kolory['Ve'], lw=1.0)
    ax10.set_xlabel('Potencjał Korelacyjny Vk', fontsize=8)
    ax10.set_ylabel('Potencjał Estymacyjny Ve', fontsize=8)
    ax10.set_title('J) Zależność Ve(Vk) z Asymptotą', fontsize=9, pad=6)
    ax10.axhline(prog_decyzyjny_vd, color='black', linestyle='-.', label=f'Próg Vd')

    # Subplot K: Zależność Ve(t) z progiem Vd
    ax11 = osie[3, 1]
    ax11.clear()
    ax11.plot(wektor_czasu, wartosci_ve, color=kolory['Ve'], lw=1.0)
    ax11.axhline(prog_decyzyjny_vd, color='black', linestyle='-.', label=f'Próg Vd')
    ax11.set_xticks(xticks)
    ax11.set_xticklabels([f'{xtick:.0f}' for xtick in xticks])
    ax11.set_xlabel(etykieta_x, fontsize=8)
    ax11.set_ylabel('Potencjał Estymacyjny Ve', fontsize=8)
    ax11.legend(loc='upper right', fontsize=7)
    ax11.set_title('K) Zależność Ve(t) z Progiem Vd', fontsize=9, pad=6)

    # Subplot L: Zależność Ve(Vk) z progiem Vd (alternatywa)
    ax12 = osie[3, 2]
    ax12.clear()
    ax12.scatter(wartosci_vk, wartosci_ve, color=kolory['Ve'], alpha=0.8)
    ax12.axhline(prog_decyzyjny_vd, color='black', linestyle='-.', label=f'Próg Vd')
    ax12.set_xlabel('Potencjał Korelacyjny Vk', fontsize=8)
    ax12.set_ylabel('Potencjał Estymacyjny Ve', fontsize=8)
    ax12.legend(loc='upper right', fontsize=7)
    ax12.set_title('L) Zależność Ve(Vk) z Progiem Vd (Alternatywa)', fontsize=9, pad=6)

    fig.canvas.draw_idle()


fig, osie = plt.subplots(4, 3, figsize=(18, 27))
plt.subplots_adjust(hspace=0.6, wspace=0.4, bottom=0.25, top=0.95)

# Położenia suwaków - uklad 3-kolumnowy, skrócone suwaki, większe odstępy - bez zmian
slider_width = 0.20
slider_height = 0.015
start_x_col1 = 0.05
start_x_col2 = start_x_col1 + slider_width + 0.10
start_x_col3 = start_x_col2 + slider_width + 0.10
start_y_row1 = 0.05
delta_y_slider = 0.03

ax_maks_czas = plt.axes([start_x_col1, start_y_row1 + 5*delta_y_slider, slider_width, slider_height])
ax_krok_czas = plt.axes([start_x_col1, start_y_row1 + 4*delta_y_slider, slider_width, slider_height])

ax_opoznienie_rej = plt.axes([start_x_col1, start_y_row1 + 3*delta_y_slider, slider_width, slider_height])
ax_opoznienie_reak = plt.axes([start_x_col1, start_y_row1 + 2*delta_y_slider, slider_width, slider_height])
ax_waga_est = plt.axes([start_x_col1, start_y_row1 + 1*delta_y_slider, slider_width, slider_height])
ax_ampl_szumu_rec = plt.axes([start_x_col1, start_y_row1 + 0*delta_y_slider, slider_width, slider_height])


ax_przew_pocz = plt.axes([start_x_col2, start_y_row1 + 5*delta_y_slider, slider_width, slider_height])
ax_alfa_g = plt.axes([start_x_col2, start_y_row1 + 4*delta_y_slider, slider_width, slider_height])
ax_ekst_rej = plt.axes([start_x_col2, start_y_row1 + 3*delta_y_slider, slider_width, slider_height])
ax_ekst_derej = plt.axes([start_x_col2, start_y_row1 + 2*delta_y_slider, slider_width, slider_height])
ax_maks_moc_efekt = plt.axes([start_x_col2, start_y_row1 + 1*delta_y_slider, slider_width, slider_height])
ax_wsp_vp_k = plt.axes([start_x_col2, start_y_row1 + 0*delta_y_slider, slider_width, slider_height])


ax_wsp_vp_delta_g = plt.axes([start_x_col3, start_y_row1 + 5*delta_y_slider, slider_width, slider_height])
ax_k_vh_vp = plt.axes([start_x_col3, start_y_row1 + 4*delta_y_slider, slider_width, slider_height])
ax_prog_decyz = plt.axes([start_x_col3, start_y_row1 + 3*delta_y_slider, slider_width, slider_height])
ax_wsp_rownowagi = plt.axes([start_x_col3, start_y_row1 + 2*delta_y_slider, slider_width, slider_height])
ax_button = plt.axes([0.40, 0.01, 0.2, 0.04])

# Suwaki
suwak_maks_czas = Slider(ax_maks_czas, 'Max Czas', 10.0, 150.0, valinit=DEFAULT_MAKSYMALNY_CZAS_SYMULACJI)
suwak_krok_czas = Slider(ax_krok_czas, 'Krok Czas', 0.01, 0.5, valinit=DEFAULT_KROK_CZASOWY_DT)

suwak_opoznienie_rej = Slider(ax_opoznienie_rej, 'τ Rej.', 0.0, 5.0, valinit=DEFAULT_OPOZNIENIE_REJESTRACJI_TAU)
suwak_opoznienie_reak = Slider(ax_opoznienie_reak, 'τ Reak.', 0.0, 5.0, valinit=DEFAULT_OPOZNIENIE_REAKCJI_TAU)
suwak_waga_est = Slider(ax_waga_est, 'Waga Est.', 0.0, 1.0, valinit=DEFAULT_WAGA_ESTYMACJI_WT)
suwak_ampl_szumu_rec = Slider(ax_ampl_szumu_rec, 'Szum Rec.', 0.0, 0.1, valinit=DEFAULT_AMPLITUDA_SZUMU_RECEPTORA)
suwak_przew_pocz = Slider(ax_przew_pocz, 'G_pocz', 0.01, 0.5, valinit=DEFAULT_PRZEWODNOSC_POCZATKOWA_G0, valstep=0.01)
suwak_alfa_g = Slider(ax_alfa_g, 'Alfa_G', 0.0, 0.1, valinit=DEFAULT_ALFA_G_PRZEWODNOSCI, valstep=0.001)
suwak_ekst_rej = Slider(ax_ekst_rej, 'Ext R', 0.0, 0.1, valinit=DEFAULT_EKSTYNKCJA_REJESTRACJI_R, valstep=0.001)
suwak_ekst_derej = Slider(ax_ekst_derej, 'Ext D', 0.0, 0.1, valinit=DEFAULT_EKSTYNKCJA_DEREJESTRACJI_D, valstep=0.001)
suwak_maks_moc_efekt = Slider(ax_maks_moc_efekt, 'Max Moc Ef.', 0.1, 2.0, valinit=DEFAULT_MAKSYMALNA_MOC_EFEKTORA)
suwak_wsp_vp_k = Slider(ax_wsp_vp_k, 'Wsp. Vp_K', 0.0, 0.2, valinit=DEFAULT_WSPOLCZYNNIK_VP_K, valstep=0.001)
suwak_wsp_vp_delta_g = Slider(ax_wsp_vp_delta_g, 'Wsp. Vp_ΔG', 0.0, 5.0, valinit=DEFAULT_WSPOLCZYNNIK_VP_DELTA_G)
suwak_k_vh_vp = Slider(ax_k_vh_vp, 'K_Vh_Vp', 0.0, 1.0, valinit=DEFAULT_K_VH_VP, valstep=0.01)
suwak_prog_decyz = Slider(ax_prog_decyz, 'Próg Vd', 0.0, 1.0, valinit=DEFAULT_PROG_DECYZYJNY_VD, valstep=0.01)
suwak_wsp_rownowagi = Slider(ax_wsp_rownowagi, 'Wsp. Wr', 0.0, 1.0, valinit=DEFAULT_WSPOLCZYNNIK_ROWNOWAGI_WR, valstep=0.01)

# Przycisk "Przegeneruj" - przesunięty w prawo i w górę, oś ukryta
ax_button = plt.axes([0.75, 0.05, 0.15, 0.04])
button = Button(ax_button, 'Przegeneruj', hovercolor='0.975')
ax_button.set_xticks([])
ax_button.set_yticks([])

def aktualizuj_wykresy():
    maks_czas = suwak_maks_czas.val
    krok_czas = suwak_krok_czas.val

    opoznienie_rej_tau = suwak_opoznienie_rej.val
    opoznienie_reak_tau = suwak_opoznienie_reak.val
    waga_est_wt = suwak_waga_est.val
    ampl_szumu_rec = suwak_ampl_szumu_rec.val
    przew_pocz_g0 = suwak_przew_pocz.val
    alfa_g_przew = suwak_alfa_g.val
    ekst_rej_r = suwak_ekst_rej.val
    ekst_derej_d = suwak_ekst_derej.val
    maks_moc_efektora = suwak_maks_moc_efekt.val
    wsp_vp_k = suwak_wsp_vp_k.val
    wsp_vp_delta_g = suwak_wsp_vp_delta_g.val
    k_vh_vp_val = suwak_k_vh_vp.val
    prog_decyz_vd = suwak_prog_decyz.val
    wsp_rownowagi_wr = suwak_wsp_rownowagi.val

    wyniki_symulacji = uruchom_symulacje(
        maksymalny_czas_symulacji=maks_czas,
        krok_czasowy_dt=krok_czas,
        opoznienie_rejestracji_tau=opoznienie_rej_tau,
        opoznienie_reakcji_tau=opoznienie_reak_tau,
        waga_estymacji_wt=waga_est_wt,
        amplituda_szumu_receptora=ampl_szumu_rec,
        przewodnosc_poczatkowa_g0=przew_pocz_g0,
        alfa_g_przewodnosci=alfa_g_przew,
        ekstynkcja_rejestracji_r=ekst_rej_r,
        ekstynkcja_derejestracji_d=ekst_derej_d,
        maksymalna_moc_efektora=maks_moc_efektora,
        wspolczynnik_vp_k=wsp_vp_k,
        wspolczynnik_vp_delta_g=wsp_vp_delta_g,
        k_vh_vp=k_vh_vp_val,
        prog_decyzyjny_vd=prog_decyz_vd,
        wspolczynnik_rownowagi_wr=wsp_rownowagi_wr
    )
    wektor_czasu, suma_bodzcow, wartosci_vr, wartosci_vh, wartosci_ve, wartosci_g, wartosci_k, wartosci_vk, wartosci_reakcji, czasy_decyzji, czasy_reakcji, wartosci_vp, wartosci_swiadomosci, wartosci_myslenia, wartosci_kg_graniczne, wartosci_delta_g, vk_decyzje = wyniki_symulacji

    generuj_wykresy_wynikow_symulacji(fig, osie, wektor_czasu, suma_bodzcow, wartosci_vr, wartosci_vh, wartosci_ve, wartosci_g, wartosci_k,
                                    wartosci_vk, wartosci_reakcji, czasy_decyzji, czasy_reakcji, wartosci_vp, wartosci_swiadomosci,
                                    wartosci_myslenia, wartosci_kg_graniczne, wartosci_delta_g, vk_decyzje, prog_decyz_vd, opoznienie_reak_tau)

def on_button_clicked(event):
    aktualizuj_wykresy()

button.on_clicked(on_button_clicked)
aktualizuj_wykresy()
plt.show()