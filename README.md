# Symulacja procesów sterowania w układach samodzielnych

Symulacja opiera się na cybernetycznym wzorcu systemu autonomicznego opracowanym przez Mariana Mazura. Więcej informacji na temat teorii systemów autonomicznych można znaleźć na stronie: [autonom.edu.pl](http://autonom.edu.pl/).

## Kluczowe funkcjonalności
- **Rejestracja i derejestracja informacji**: Przechowywanie i zanikanie danych w zależności od aktualności i trwałości bodźców.
- **Sterowanie adaptacyjne**: Mechanizmy korelacji danych teraźniejszych i przeszłych w celu podejmowania decyzji.
- **Estymacja i podejmowanie decyzji**: Implementacja progów decyzyjnych oraz czasu reakcji w systemach korelacyjnych.
- **Modelowanie matematyczne**:
  - Eksponencjalny zanik rejestratów.
  - Dynamika potencjałów korelacyjnych i refleksyjnych.
  - Równania różniczkowe opisujące procesy sterowania.

## Przykłady zastosowań
- Symulacja mechanizmów decyzyjnych w układach adaptacyjnych.
- Modele procesów pamięciowych w systemach biologicznych.
- Układy regulacji temperatury

## Instrukcje uruchamiania

1. **Uruchomienie symulacji**
   Skrypt `simulation.py` należy uruchomić jako pierwszy, aby wygenerować dane wejściowe dla pozostałych modułów:
   ```bash
   python symulacja/simulation.py
   ```

2. **Uruchamianie pozostałych modułów**
   Po wygenerowaniu danych przez `simulation.py`, możesz niezależnie uruchomić dowolny z pozostałych modułów:
   - `analiza.py`: Analiza wyników symulacji
     ```bash
     python symulacja/analiza.py
     ```
   - `corrector.py`: Moduł korelatora
     ```bash
     python symulacja/corrector.py
     ```
   - `effector.py`: Moduł efektora
     ```bash
     python symulacja/effector.py
     ```
   - `homeostat.py`: Moduł homeostatyczny
     ```bash
     python symulacja/homeostat.py
     ```
   - `receptor.py`: Moduł receptora
     ```bash
     python symulacja/receptor.py
     ```

3. **Środowisko wirtualne**
   Zaleca się korzystanie ze środowiska wirtualnego. Aby je skonfigurować:
   ```bash
   python -m venv venv
   source venv/bin/activate  # lub `venv\Scripts\activate` na Windows
   pip install -r requirements.txt
   ```

## Wymagania
Lista zależności znajduje się w pliku `requirements.txt`. Aby je zainstalować, uruchom:
```bash
pip install -r requirements.txt
```
