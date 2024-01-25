# USD_23Z

### Projekt końcowy z przedmiotu [USD] - Uczące się Systemy Decyzyjne
**Temat**: </br>
*LL10. rl-baselines3-zoo* </br>
Zapoznaj się z biblioteką *rl-baselines3-zoo* (https://rl-baselines3-zoo.readthedocs.io/en/master).
Wybierz co najmniej 2 kombinacje środowisko+algorytm, dla których nie są dostępne zoptymalizowane hiperparametry (w miarę możliwości środowiska powinny być z różnych kategorii). Dokonaj optymalizacji hiperparametrów. Przedstaw wpływ hiperparametrów algorytmu na zachowanie wytrenowanego agenta (opisowo oraz - w miarę możliwości - wizualnie).

## Instalacja i Uruchomienie

1. Upewnij się, ze na twojej maszynie zainstalowany jest Python 3 (najlepiej w wersji 3.9) 
2. Sklonuj repozytorium:
```bash
$ git clone https://github.com/kucjan/USD_23Z-Kuc-Bugala.git
```
2. Przejdz do katalogu projektowego.
3. Stworz i aktywuj wirtualne środowisko:
```bash
$ python3 -m venv venv
$ source venv/bin/activate
```
4. Zainstaluj wymagane biblioteki:
```bash
$ pip3 install rl_zoo3
$ pip3 install -r requirements_rl_zoo3.txt
```
5. W celu uruchomienia skryptu pozwalającego na odtworzenie dowolnego eksperymentu zawartego w projekcie, najpierw przejdz do katalogu *reproduce*:
```bash
$ cd reproduce
```
6. Będąc w tym katalogu wywołaj polecenie (w miejsce **{exp_id}** wstaw numer eksperymentu):
- dla środowiska **InvertedDoublePendulumBulletEnv-v0** oraz algorytmu **DDPG**:
```bash
$ python3 reproduce.py ddpg InvertedDoublePendulumBulletEnv-v0 {exp_id}
```
- dla środowiska **LunarLander-v2** oraz algorytmu **ARS**:
```bash
$ python3 reproduce.py ars LunarLander-v2 {exp_id}
```
W wyniku działania skryptu w konsoli wyświetli się zestaw hiperparametrów uzytych do treningu, wyniki ewaluacji najlepszego modelu, a następnie na ekranie zaprezentowane zostaną wyniki ewaluacji przeprowadzanej w trakcie treningu w formie wykresów. Na koniec wyświetlone zostanie działanie wytrenowanego agenta w formie graficznej.
Istnieje takze opcja odtworzenia całego treningu dla wybranego eksperymentu, przy uzyciu tego samego ziarna (*seed*). Wystarczy do polecenia, na końcu dopisać **train**. Przykładowo:
```bash
$ python3 reproduce.py ddpg InvertedDoublePendulumBulletEnv-v0 {exp_id} train
```
8. Kiedy skończysz pracować nad projektem wywołaj:
```bash
$ deactivate
```
