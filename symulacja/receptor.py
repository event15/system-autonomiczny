import numpy as np
import collections


class Receptor:
    def __init__(self, nazwa="Receptor", amplituda_szumu=0.05, prog_detekcji_bodzca=0.1, czas_adaptacji_tau_a=3.0,
                 opoznienie_rejestracji_tau_s=0.2):
        self.nazwa = nazwa
        self.amplituda_szumu = amplituda_szumu
        self.prog_detekcji_bodzca = prog_detekcji_bodzca
        self.czas_adaptacji_tau_a = czas_adaptacji_tau_a
        self.opoznienie_rejestracji_tau_s = opoznienie_rejestracji_tau_s
        self.Vr = 0.0
        self.potencjal_bazowy = 0.0
        self.historia_bodzcow = collections.deque(maxlen=int(
            opoznienie_rejestracji_tau_s / 0.1) if opoznienie_rejestracji_tau_s > 0 else 1)

    def procesuj_bodziec(self, wartosc_bodzca, aktualny_czas, krok_czasowy_dt=0.1):
        self.historia_bodzcow.append(wartosc_bodzca)
        if len(self.historia_bodzcow) < self.historia_bodzcow.maxlen:
            bodziec_opozniony = 0.0
        else:
            bodziec_opozniony = self.historia_bodzcow[0]

        bodziec_po_progu = max(0, bodziec_opozniony - self.prog_detekcji_bodzca)

        szum = self.amplituda_szumu * np.random.normal()

        czas_staly_adaptacji = self.czas_adaptacji_tau_a
        if czas_staly_adaptacji > 0:
            self.potencjal_bazowy += (bodziec_po_progu - self.potencjal_bazowy) * (
                    1 - np.exp(-krok_czasowy_dt / czas_staly_adaptacji))
        else:
            self.potencjal_bazowy = bodziec_po_progu

        self.Vr = self.potencjal_bazowy + szum

        return self.Vr

    def pobierz_potencjal(self):
        return self.Vr


if __name__ == '__main__':
    receptor1 = Receptor(nazwa="ReceptorAdaptacyjny", amplituda_szumu=0.02, prog_detekcji_bodzca=0.3,
                         czas_adaptacji_tau_a=2.0, opoznienie_rejestracji_tau_s=0.5)
    receptor2 = Receptor(nazwa="ReceptorProsty", amplituda_szumu=0.05, prog_detekcji_bodzca=0.0,
                         czas_adaptacji_tau_a=0.0,
                         opoznienie_rejestracji_tau_s=0.0)

    punkty_czasowe = np.arange(0, 30, 0.1)
    wartosci_vr1 = []
    wartosci_vr2 = []

    for i, t in enumerate(punkty_czasowe):
        bodziec1 = 0.7 if 5 < t < 20 else 0.2
        bodziec2 = bodziec1

        vr1 = receptor1.procesuj_bodziec(bodziec1, t)
        vr2 = receptor2.procesuj_bodziec(bodziec2, t)

        wartosci_vr1.append(vr1)
        wartosci_vr2.append(vr2)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(punkty_czasowe, wartosci_vr1, label=f'{receptor1.nazwa} (Vr) - Adaptacyjny, z Progiem i Opóźnieniem')
    plt.plot(punkty_czasowe, wartosci_vr2, label=f'{receptor2.nazwa} (Vr) - Prosty (dla porównania)')
    plt.plot(punkty_czasowe, [receptor1.prog_detekcji_bodzca] * len(punkty_czasowe), '--r',
             label='Próg Detekcji Receptora Adaptacyjnego')
    plt.plot(punkty_czasowe, [bodziec1] * len(punkty_czasowe), '--k', label='Bodziec')

    plt.xlabel("Czas")
    plt.ylabel("Potencjał Receptora (Vr) / Bodziec")
    plt.title("Porównanie Receptorów: Adaptacyjny vs. Prosty, z Progiem Detekcji i Opóźnieniem")
    plt.legend()
    plt.grid(True)
    plt.show()
