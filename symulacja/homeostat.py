import numpy as np

class Homeostat:
    def __init__(self, nazwa="Homeostat", wspolczynnik_rownowagi_wr=0.5, prog_decyzyjny_vd=0.6, k_vh_vp=0.5):
        self.nazwa = nazwa
        self.wspolczynnik_rownowagi_wr = wspolczynnik_rownowagi_wr
        self.prog_decyzyjny_vd = prog_decyzyjny_vd
        self.k_vh_vp = k_vh_vp
        self.potencjal_refleksyjny_vh = 0.0
        self.potencjal_perturbacyjny_vp = 0.0

    def ustaw_potencjal_refleksyjny_proporcjonalnie(self):
        self.potencjal_refleksyjny_vh = self.k_vh_vp * self.potencjal_perturbacyjny_vp
        return self.potencjal_refleksyjny_vh

    def pobierz_potencjal_refleksyjny(self):
        return self.potencjal_refleksyjny_vh

    def ustaw_potencjal_perturbacyjny(self, wartosc_vp):
        self.potencjal_perturbacyjny_vp = wartosc_vp
        return self.potencjal_perturbacyjny_vp

    def pobierz_potencjal_perturbacyjny(self):
        return self.potencjal_perturbacyjny_vp


if __name__ == '__main__':
    homeostat = Homeostat()
    punkty_czasowe = np.arange(0, 20, 0.1)
    wartosci_vh = []
    wartosci_ve_przyklad = []

    for t in punkty_czasowe:
        wartosc_ve = 0.7 if 5 < t < 15 else 0.3
        wartosci_ve_przyklad.append(wartosc_ve)
        wartosci_vh.append(homeostat.pobierz_potencjal_refleksyjny())

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(punkty_czasowe, wartosci_vh, label='Potencjał Refleksyjny Vh(t)')
    plt.plot(punkty_czasowe, wartosci_ve_przyklad, linestyle='--', label='Przykładowy Potencjał Estymacyjny Ve(t)')
    plt.xlabel("Czas")
    plt.ylabel("Wartość Potencjału")
    plt.title("Homeostat - Dynamika Potencjału Refleksyjnego")
    plt.legend()
    plt.grid(True)
    plt.show()