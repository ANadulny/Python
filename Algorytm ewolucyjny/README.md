### Zadanie:

Stworzyć program, który za pomocą algorytmu ewolucyjnego zaplanuje rozmieszczenie 
szpitalnych oddziałów udarowych w Polsce przy założeniu, że oddzial może znajdować się w 
mieście o populacji powyżej 60 tysięcy mieszkańców (z dnia 01.01.2018 r.), każdy pacjent w 
Polsce powinien być dowieziony do takiego oddziału w czasie nie większym niż 150 minut, a 
średnia prędkość karetki pogotowia to 75 km/h. Program powinien minimalizować liczbe 
miast, w ktorych znajdują sie oddzialy udarowe. Interfejs programu powinien umozliwiać 
graficzną prezentacje wyniku.


Co powinien zawierać projekt ostateczny (6-10 stron):
1. to co projekt wstępny, ewentualnie poprawione i uzupełnione 
2. pełen opis funkcjonalny 
3. opis interfejsu użytkownika 
4. postać plików konfiguracyjnych, logów, itp. 
5. raport z przeprowadzonych testów oraz wnioski 
6. opis wykorzystanych narzędzi, bibliotek, itp 
7. opis metodyki testów i wyników testowania.

---

# How to use

1. Install pipenv (and optionally pyenv if you don't have required python version)
2. Go into the root directory of this project
3. Create virtual environment and install needed packages:
    ```
    pipenv install
    ```
4. (optional) Export environment variable `GOOGLE_API_KEY` (or put it in `.env` file) with your google maps API key
5. Run the code:
    ```
    pipenv run python main.py
    ```
