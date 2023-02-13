# Próbafeladat

Rendelkezésre áll egy nagyon egyszerű, ámde meglehetősen hiányos script,
amely egy borok minőségének meghatározására alkalmas 
ML modellt hivatott tanítani és kiértékelni.

A feladat ezen megoldás kiegészítése az alábbi követelményeknek eleget téve.

### Követelmények
- Képes legyen az eredmények megjelenítésére és mentésére (lásd: Eredmények megjelenítése).
- Legyen lehetőség teszthalmaz elkülönítésére 
- Az alábbiak legyenek konfigurálhatóak:
    - Az adatfájl elérési útvonala, az eredmények mentési helye.
    - A teszthalmazba tartozó elemek aránya
    - A tanításhoz használt jellemzők.

       _Például legyen lehetőség csak a pH és alkohol paramétereken tanítani._ 
    - A modell argumentumai (a modell típusán nem kell változtatni).
  
       _Például legyen lehetőség a modell kernelét rbf típusúra állítani._
- Legyen parancssorból kényelmesen konfigurálható és indítható.
- Legyen a megoldás jól tagolt, átlátható.

### Eredmények megjelenítése
Az eredmények megjelenítése alapvetően a feladat megoldójára van bízva. 
Az elvárás annyi, hogy egy könnyen átlátható és értelmezhető ábra készüljön el, amiből látszik, hogy a modell kimenet mennyire megfelelő.
Az alkalmazott könyvtár lehet például _matplotlib_ vagy _seaborn_, de egyéb tetszőleges 
könyvtár is alkalmazható.

### Opcionális feladatrész
Plusz pontot ér, amennyiben az elkészült megoldás Docker konténerizált.

### Fontos
A feladat nem egy jó ML modell elkészítése, 
hanem egy általános megoldás implementálása, amely segítségével könnyebben lehet
különböző kísérleteket végezni és azok eredményeit kiértékelni. 

A legfontosabb szempont, hogy megfelel-e a fenti követelményeknek,
továbbá mennyire jól struktúrált, átlátható, igényes, illetve a nyelvi elemeket 
alkalmazó az elkészült megoldás.

A feladat megoldása során lehet alkalmazni tetszőleges, további hasznos könyvtárakat 
amelyek emelik a végeredmény színvonalát.
