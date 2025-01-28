import numpy as np


class Efektor:
    def __init__(self, nazwa="Efektor", Vd=0.6, opoznienie_reakcji_tau=2.5, czas_trwania_reakcji_tau_r=5.0, P_max=1.0):
        self.nazwa = nazwa
        self.Vd = Vd
        self.opoznienie_reakcji_tau = opoznienie_reakcji_tau
        self.czas_trwania_reakcji_tau_r = czas_trwania_reakcji_tau_r
        self.P_max = P_max
        self.reakcja = 0.0
        self.czas_start_reakcji = -np.inf
        self.kroki_opoznienia_reakcji = int(opoznienie_reakcji_tau / 0.1)
        self.zaplanowane_reakcje = []

    def procesuj_decyzje(self, Ve, aktualny_czas, ostatnia_reakcja):
        decyzja = False

        if Ve > self.Vd and ostatnia_reakcja == 0:
            decyzja = True

        if decyzja:
            czas_zaplanowanej_reakcji = aktualny_czas + self.opoznienie_reakcji_tau
            indeks_zaplanowanej_reakcji = int(czas_zaplanowanej_reakcji / 0.1)
            self.zaplanowane_reakcje.append({'czas': czas_zaplanowanej_reakcji, 'indeks': indeks_zaplanowanej_reakcji})
            return True

        return False

    def wykonaj_reakcje(self, aktualny_czas, wartosc_Ve, zaplanowane_reakcje, wartosci_reakcji, Vd, czasy_reakcji,
                        wektor_czasu):
        reakcje_do_wykonania = []
        nastepne_zaplanowane_reakcje = []

        for zaplanowana_reakcja in zaplanowane_reakcje:
            if zaplanowana_reakcja['czas'] <= aktualny_czas:
                if wartosc_Ve > Vd:
                    reakcje_do_wykonania.append(zaplanowana_reakcja)
                else:
                    print(f"Reakcja ANULOWANA w czasie {aktualny_czas:.2f}s, Ve spadło poniżej Vd.")
                    pass
            else:
                nastepne_zaplanowane_reakcje.append(zaplanowana_reakcja)

        for reakcja in reakcje_do_wykonania:
            indeks_reakcji = reakcja['indeks']
            if 0 <= indeks_reakcji < len(wartosci_reakcji):
                wartosci_reakcji[indeks_reakcji] = 1
                czasy_reakcji.append(wektor_czasu[indeks_reakcji])
                print(f"Reakcja WYKONANA w czasie {aktualny_czas:.2f}s, Ve={wartosc_Ve:.3f} > Vd={Vd:.3f}")
                self.czas_start_reakcji = aktualny_czas

        self.zaplanowane_reakcje = nastepne_zaplanowane_reakcje

        # Derejestracja reakcji - czas trwania reakcji
        if self.reakcja == 1 and aktualny_czas > self.czas_start_reakcji + self.czas_trwania_reakcji_tau_r:
            indeks_aktualnego_czasu = int(aktualny_czas / 0.1)
            if 0 <= indeks_aktualnego_czasu < len(wartosci_reakcji):
                wartosci_reakcji[indeks_aktualnego_czasu] = 0
                print(f"Derejestracja reakcji w czasie {aktualny_czas:.2f}s, czas trwania reakcji upłynął.")
                self.reakcja = 0
                self.czas_start_reakcji = -np.inf
            else:
                self.reakcja = 0

        indeks_aktualnego_czasu = int(aktualny_czas / 0.1)
        if 0 <= indeks_aktualnego_czasu < len(wartosci_reakcji):
            if self.reakcja != 0:
                self.reakcja = wartosci_reakcji[indeks_aktualnego_czasu]
            elif wartosci_reakcji[indeks_aktualnego_czasu] == 1:
                self.reakcja = wartosci_reakcji[indeks_aktualnego_czasu]
            else:
                self.reakcja = 0
        else:
            self.reakcja = 0

    def pobierz_reakcje(self):
        return self.reakcja

    def pobierz_pobor_mocy(self):
        return self.P_max if self.reakcja == 1 else 0.0


if __name__ == '__main__':
    efektor = Efektor(opoznienie_reakcji_tau=2.0, czas_trwania_reakcji_tau_r=7.0)
    punkty_czasowe = np.arange(0, 30, 0.1)
    wartosci_ve_przyklad = []
    wartosci_reakcji = np.zeros_like(punkty_czasowe)
    zaplanowane_reakcje_test = []

    czasy_reakcji_test = []
    wektor_czasu_test = punkty_czasowe
    prog_decyzyjny_vd_test = 0.6

    for i, t in enumerate(punkty_czasowe):
        wartosc_ve = 0.7 if 5 < t < 15 else 0.3
        wartosci_ve_przyklad.append(wartosc_ve)
        sygnal_decyzji = efektor.procesuj_decyzje(wartosc_ve, t, wartosci_reakcji[i - 1] if i > 0 else 0)

        efektor.wykonaj_reakcje(t, wartosc_ve, efektor.zaplanowane_reakcje, wartosci_reakcji, Vd=prog_decyzyjny_vd_test,
                                czasy_reakcji=czasy_reakcji_test, wektor_czasu=wektor_czasu_test)

    wyjscie_reakcji_efektora = []
    for r in wartosci_reakcji:
        wyjscie_reakcji_efektora.append(efektor.pobierz_reakcje())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(punkty_czasowe, wartosci_ve_przyklad, linestyle='--', label='Potencjał Estymacyjny Ve(t)')
    plt.step(punkty_czasowe, wartosci_reakcji, where='post', label='Reakcja Efektora (t)')
    plt.xlabel("Czas")
    plt.ylabel("Wartość")
    plt.title("Efektor - Reakcja z Derejestracją Czasową")
    plt.legend()
    plt.grid(True)
    plt.show()
