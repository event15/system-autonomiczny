import numpy as np
from homeostat import Homeostat


class Korelator:
    def __init__(self, nazwa="Korelator", przewodnosc_poczatkowa_g0=0.1, alfa_g=0.05, ekstynkcja_r=0.02,
                 ekstynkcja_d=0.01, wspolczynnik_vp_k=0.1, wspolczynnik_vp_delta_g=1.0):
        self.nazwa = nazwa
        self.przewodnosc_g = przewodnosc_poczatkowa_g0
        self.przewodnosc_poczatkowa_g0 = przewodnosc_poczatkowa_g0
        self.alfa_g = alfa_g
        self.ekstynkcja_rejestracji_r = ekstynkcja_r
        self.ekstynkcja_derejestracji_d = ekstynkcja_d
        self.moc_korelacyjna_k = 0.0
        self.potencjal_estymacyjny_ve = 0.0
        self.potencjal_korelacyjny_vk = 0.0
        self.wspolczynnik_vp_k = wspolczynnik_vp_k
        self.wspolczynnik_vp_delta_g = wspolczynnik_vp_delta_g
        self.poprzednia_przewodnosc_g = przewodnosc_poczatkowa_g0
        self.kg_graniczne = 0.0 # Dodano atrybut dla granicznej mocy korelacyjnej

    def oblicz_potencjal_korelacyjny(self, potencjal_rejestracyjny_vr_suma, potencjal_refleksyjny_vh):
        self.potencjal_korelacyjny_vk = potencjal_rejestracyjny_vr_suma + potencjal_refleksyjny_vh
        return self.potencjal_korelacyjny_vk

    def oblicz_moc_korelacyjna(self):
        self.moc_korelacyjna_k = self.potencjal_korelacyjny_vk * self.przewodnosc_g
        return self.moc_korelacyjna_k

    def oblicz_potencjal_estymacyjny(self, waga_wt=0.8):
        self.potencjal_estymacyjny_ve = waga_wt * self.moc_korelacyjna_k
        return self.potencjal_estymacyjny_ve

    def aktualizuj_przewodnosc_rejestracja(self, wartosc_bodzca, dt):
        self.kg_graniczne = self.oblicz_kg_graniczne() # Oblicz i zapisz kg_graniczne
        gg = self.oblicz_gg_graniczne()
        g_docelowe = gg - (gg - self.przewodnosc_poczatkowa_g0) * np.exp(-self.ekstynkcja_rejestracji_r * dt)
        dg = (g_docelowe - self.przewodnosc_g)
        self.przewodnosc_g += dg

    def aktualizuj_przewodnosc_derejestracja(self, dt):
        self.kg_graniczne = self.oblicz_kg_graniczne() # Oblicz i zapisz kg_graniczne
        gp = self.przewodnosc_g
        g_docelowe = self.przewodnosc_poczatkowa_g0 + (gp - self.przewodnosc_poczatkowa_g0) * np.exp(
            -self.ekstynkcja_derejestracji_d * dt)
        dg = (g_docelowe - self.przewodnosc_g)
        self.przewodnosc_g += dg

    def oblicz_kg_graniczne(self):
        mianownik = 1 - self.alfa_g * self.potencjal_korelacyjny_vk * self.przewodnosc_poczatkowa_g0
        if mianownik == 0:
            return float('inf')
        return (self.potencjal_korelacyjny_vk * self.przewodnosc_poczatkowa_g0) / mianownik

    def oblicz_gg_graniczne(self):
        mianownik = 1 - self.alfa_g * self.potencjal_korelacyjny_vk * self.przewodnosc_poczatkowa_g0
        if mianownik == 0:
            return float('inf')
        return self.przewodnosc_poczatkowa_g0 / mianownik

    def pobierz_przewodnosc(self):
        return self.przewodnosc_g

    def pobierz_moc_korelacyjna(self):
        return self.moc_korelacyjna_k

    def pobierz_potencjal_estymacyjny(self):
        return self.potencjal_estymacyjny_ve

    def pobierz_potencjal_korelacyjny_vk(self):
        return self.potencjal_korelacyjny_vk

    def pobierz_kg_graniczne(self): # Dodano pobieranie kg_graniczne
        return self.kg_graniczne

    def aktualizuj_homeostat_vp(self, homeostat):
        delta_g = abs(self.przewodnosc_g - self.poprzednia_przewodnosc_g)
        wartosc_vp = (self.moc_korelacyjna_k * self.wspolczynnik_vp_k) + (
                    delta_g * self.wspolczynnik_vp_delta_g)
        homeostat.ustaw_potencjal_perturbacyjny(wartosc_vp)
        self.poprzednia_przewodnosc_g = self.przewodnosc_g


if __name__ == '__main__':
    homeostat = Homeostat()
    korelator = Korelator()

    punkty_czasowe = np.arange(0, 20, 0.1)
    wartosci_g = []
    wartosci_k = []
    wartosci_ve = []
    wartosci_vk = []
    wartosci_vh = []
    wartosci_vp = []
    wartosci_delta_g = []
    wartosci_kg_graniczne = [] # Dodano listę dla kg_graniczne

    for t in punkty_czasowe:
        wartosc_bodzca = 1.0 if 5 < t < 15 else 0.0
        potencjal_rejestracyjny_vr_suma = wartosc_bodzca

        wartosc_vh = homeostat.pobierz_potencjal_refleksyjny()

        vk = korelator.oblicz_potencjal_korelacyjny(potencjal_rejestracyjny_vr_suma, wartosc_vh)
        k = korelator.oblicz_moc_korelacyjna()
        ve = korelator.oblicz_potencjal_estymacyjny()

        poprzednia_g = korelator.pobierz_przewodnosc()
        if wartosc_bodzca > 0:
            korelator.aktualizuj_przewodnosc_rejestracja(wartosc_bodzca, 0.1)
        else:
            korelator.aktualizuj_przewodnosc_derejestracja(0.1)
        delta_g = abs(korelator.pobierz_przewodnosc() - poprzednia_g)

        korelator.aktualizuj_homeostat_vp(homeostat)
        homeostat.ustaw_potencjal_refleksyjny_proporcjonalnie()

        wartosci_g.append(korelator.pobierz_przewodnosc())
        wartosci_k.append(korelator.pobierz_moc_korelacyjna())
        wartosci_ve.append(korelator.pobierz_potencjal_estymacyjny())
        wartosci_vk.append(korelator.pobierz_potencjal_korelacyjny_vk())
        wartosci_vh.append(homeostat.pobierz_potencjal_refleksyjny())
        wartosci_vp.append(homeostat.pobierz_potencjal_perturbacyjny())
        wartosci_delta_g.append(delta_g)
        wartosci_kg_graniczne.append(korelator.pobierz_kg_graniczne()) # Rejestruj kg_graniczne

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 7))
    plt.plot(punkty_czasowe, wartosci_g, label='Przewodność G(t)')
    plt.plot(punkty_czasowe, wartosci_k, label='Moc Korelacyjna K(t)')
    plt.plot(punkty_czasowe, wartosci_ve, label='Potencjał Estymacyjny Ve(t)')
    plt.plot(punkty_czasowe, wartosci_vk, label='Potencjał Korelacyjny Vk(t)')
    plt.plot(punkty_czasowe, wartosci_vh, label='Potencjał Refleksyjny Vh(t)')
    plt.plot(punkty_czasowe, wartosci_vp, label='Potencjał Perturbacyjny Vp(t)')
    plt.plot(punkty_czasowe, wartosci_delta_g, label='Zmiana Przewodności |Delta G|(t)')
    plt.plot(punkty_czasowe, wartosci_kg_graniczne, label='Graniczna Moc Korelacyjna Kg(t)', linestyle='--') # Wykres Kg

    plt.xlabel("Czas")
    plt.ylabel("Wartość")
    plt.title("Dynamika Sprzężenia Zwrotnego Homeostat - Korelator (Vp = f(K, Delta G))")
    plt.legend()
    plt.grid(True)
    plt.show()