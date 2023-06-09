﻿# Babilon-of-Machine-Learning

## Opis modelowanego zadania
W ramach projektu postanowiliśmy stworzyć program, który pomógłby w planowaniu dostaw do magazynów.
W magazynach towar nie powinien za długo zalegać, a także dopuścić do sytuacji, w której zabraknie towaru dla klienta. W tym celu postanowiliśmy stworzyć model, który starałby się przewidzieć zapotrzebowanie na produkty w przyszłości, dzięki temu można zoptymalizować łańcuchy dostaw oraz zminimalizować czas zalegania produktu w magazynie.<br>

Ze względu na skomplikowanie zadania postanowiliśmy rozbić je na mniejsze podproblemy, które są łatwiejsze w zamodelowaniu i trenowaniu. Zamiast tworzyć model, który będzie przyjmował dane ze wszystkich magazynów oraz o wszystkich produktach, postanowiliśmy skupić się na stworzeniu modelu, który przewidzi zapotrzebowanie na konkretny produkt w konkretnym magazynie. Podejście to jest prostsze do zrealizowania na naszych komputerach i stosunkowo proste do rozszerzenia na większą skalę w przypadku podejścia z większa mocą obliczeniową.

## Dataset

### Użyty dataset
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

### Opis datasetu wraz z wykorzystanymi wartościami
Wykorzystany do zadania dataset zawiera dane dotyczące sprzedaży ponad 3000 różnych produktów w 10 sklepach rozlokowanych w 3 stanach USA. Dane dokładnie opisują, ile jednostek każdego produktu zostało sprzedanych każdego dnia w każdym sklepie. Dataset zawiera również opisy ceny produktów w różnych dniach; niestety nie są to ceny dla każdego dnia, przez co postanowiliśmy zrezygnować z wykorzystania tego atrybutu. Dataset zawiera również kalendarz dni, w których były sprzedawane produkty, dokładnie opisujące, który to dzień tygodnia, który to miesiąc, czy w tym dniu odbywały się jakieś wydarzenia oraz czy w danym dniu oraz sklepie był uruchamiany program pomocy żywnościowej SNAP.<br>

Dokładniejsze wnioski z analizy datasetu są przedstawione na prezentacji

## Wybrane podejścia
Z wybranego zbioru danych wyestrahowaliśmy dane dotyczące sprzedaży produktów we wszystkich dnia dla danego sklepu, a następnie podzieliliśmy je na 30 elementowe ramki danych. Przyjęliśmy bardzo uproszczony model, lecz w łatwy sposób można dodać dodatkowe dane opisujące produkty oraz dni.<br>

W poszukiwaniu najlepszego modelu przetestowaliśmy następujące modele
- XGBoost
- Las Losowy
- Prostą sieć neuronową
- Sieć LSTM

Niestety zarówno prosta sieć neuronowa, jak i sieć LSTM nie była w stanie nauczyć się zależności. Najprawdopodobniej była to wina zbyt mało dokładnych danych użytych do uczenia, niestety nasze komputery nie pozwalały na testowanie ani bardziej rozbudowanych modeli, ani bardziej rozbudowanych danych, lecz uważamy, że jeśliby bardziej rozbudować model LSTM oraz dostarczyć mu dane bardziej rozbudowane dane, model ten byłby w stanie nauczyć się rozpoznawać podstawowe zależności.<br>

Lepsze wyniki dostaliśmy dla modeli Lasu Losowego oraz XGBoosta, ponieważ ich predykcje były bardziej zbliżone do oczekiwanych wartości.


## Możliwe rozwinięcia projektu
Dla modeli XGBoost oraz Lasu Losowego:
- Dokładniejsze dopasowanie hiperparametrów dla każdego produktu i sklepu
- Stworzenie zbioru modeli, które w sumie przewidują zapotrzebowanie dla wszystkich sklepów lub wszystkich produktów
- Zwiększenie ilości atrybutów wejściowych:
    - Dodatkowe informacje o dniach, produktach oraz sklepach
    - Zwiększenie ramki jednej sekwencji danych, aby model brał więcej niż 30 dni wstecz

Dla modeli bazujących na sieciach neuronowych:
- Rozbudowanie modeli, aby zawierały wiecej parameptrów oraz wag
- Stworzenie sieci, przewidujących zapotrzebowanie dla kilku sklepów lub kilku produktów (pomiędzy kupowanymi produktami mogą występować zależności)
- Zwiększenie ilości atrybutów wejściowych, tak jak dla modeli XGBoost oraz Lasu Losowego
