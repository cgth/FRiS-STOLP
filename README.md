# FRiS-STOLP

**Функция конкурентного сходства или FRiS-функция** — мера схожести двух объектов относительно некоторого третьего объекта. В отличие от классических мер расстояния, эта функция позволяет не просто сказать, похожи объекты друг на друга или нет, но и уточнить ответ на вопрос «по сравнению с чем?». Это позволяет большее количество факторов при классификации.

Допустим задано множество точек (назовём **M**) на плоскости, каждая из точек имеет 2 координаты.  
Возмём из него 3 точки **a, b, x** то есть **a, b, x ∈ M**

Возмём метрику **p(a,b)** - стандартная метрика - расстояние между двумя точками **a** и **b**. 
Функция конкурентного сходства объектов a, b ∈ M относительно x ∈ M задается так:

FRiS(a,b,x)=(p(a,x)-p(a,b))/(p(a,x)+p(a,b))

Легко заметить, что значения функции лежат в интервале [–1, 1], причем FRiS(a, a, x) = 1 и FRiS(a, b, a) = –1, В случае же равенства расстояний ρ(a, x) и ρ(b, x) функция примет значение 0.

Програмная реализация.
На картинке видны точки с заданной классификацией (обозначенные красным и синим цветом).
Нужно определить к какому класу принадлежит новая точка, обозначенная квадратом. Несмотря на то что она находится ближе к синим точкам, по структуре это типичный представитель класса красных точек. Программа Окрашивает квадрат красным цветом.

![f](https://user-images.githubusercontent.com/33224690/34342804-b5261e86-e970-11e7-92d4-6f413effc382.png)


**Алгоритм FRiS-СТОЛП (FRiS-STOLP)** - алгоритм отбора эталонных объектов для метрического классификатора на основе FRiS-функции. 
### Ссылки
1. http://www.pvsm.ru/pesochnitsa/10682
2. http://www.machinelearning.ru/wiki/index.php?title=FRiS-%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F
3. http://www.machinelearning.ru/wiki/index.php?title=%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%A1%D0%A2%D0%9E%D0%9B%D0%9F


