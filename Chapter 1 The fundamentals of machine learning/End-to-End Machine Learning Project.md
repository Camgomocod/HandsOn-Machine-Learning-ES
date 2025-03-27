# End-to-End Machine Learning Project

**Creación:** 2025-03-18 08:56
[[Machine Learning]]

 Pasos para un proyecto completo de ML:
1. Mirar el panorama completo.
2. Obtener los datos 
3. Descubrir y visualizar para ganar conocimientos 
4. Preparar los datos para ML algorithms 
5. Seleccionar el modelo y entrenarlo 
6. Fine-tune tu modelo
7. Presentar la solución 
8. Lazar, monitorear, y mantener tu sistema. 

## Working with real data: 

Los siguientes son links donde podemos obtener data sets con los cuales poder practicar y expandir los conocimientos prácticos. 

• Popular open data repositories
- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)
- [Kaggle datasets](https://www.kaggle.com/datasets)
- [Amazon’s AWS datasets](https://registry.opendata.aws/)
• Meta portals (they list open data repositories)
- [Data Portals](https://dataportals.org/)
- [OpenDataMonitor](https://opendatamonitor.eu/frontend/web/index.php?r=dashboard%2Findex)
- [Quandl](https://data.nasdaq.com/institutional-investors)
• Other pages listing many popular open data repositories
- [Wikipedia’s list of Machine Learning datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
- [Quora.com](https://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public)
- [The datasets subreddit](https://www.reddit.com/r/datasets/?rdt=43519)

## Data set California census 1990

![[Pasted image 20250318090750.png]]

>[!Note] 
>Usualmente se sigue la siguiente estructura de check list para organizar todo el proceso de desarrollo del modelo de ML [[Check List ML]] Hay que adaptarlo a las necesidades de cada proyecto.

## Frame the problem 

La primera pregunta que tenemos que hacer a nuestro jefe es cual es el objetivo. El construir el modelo probablemente no sea el objetivo. Cómo la compañía espera tener beneficios del modelo? Saber el objetivo es importante porque será determinante para enfoque del proyecto, que algoritmo se va a seleccionar, que medida de rendimiento se usara, y que tanto esfuerzo se gastara en modificaciones. 

El objetivo es (Una media del precio por distrito de casas) con el que se alimentara otro ML system, con otras señales. Este downstream system será determinante en si es o no gastar tiempo en un area. Tener esto claro es crítico, porque afecta directamente las ganancias. 

![[Pasted image 20250318093548.png]]

>[!Note] Pipelines
>Una secuencia de componentes de datos es llamado un data *pipeline*. Pipelines son bastante comunes en ML systems, desde que hay muchos datos que manipular y muchas transformaciones de datos que aplicar. 
>Los componentes típicamente corren asincrónicamente. Cada componente empuja una larga cantidad de datos, los procesa, y escupe el resultado en otra base de datos. Entonces, tiempo después el siguiente componente en el pipeline tira estos datos y escupe su propio output. Cada componente es bastante autónomo: Esta interfaz entre los componentes es simplemente una base de datos. Esto hace al sistema simple de comprender. (Con la ayuda del flujo de datos del gráfico), y diferentes equipos se pueden ocupar en los diferentes componentes. Además, si un componente se rompe, el rio abajo de componentes usualmente pueden continuar funcionando normalmente (Por lo menos por un rato) Usando el ultimo output del componente roto. Esto hace la arquitectura robusta. 
>Por otro lado, un componente roto puede pasar desapercibido por algún tiempo si un adecuado monitoreo no es implementado. Los datos se vuelven obsoletos y el sistema general falla. 



La siguiente pregunta que se le hace al jefe es como la solución actual luce. La situación actual a menudo dará la referencia de rendimiento, así como las ideas de como resolver el problema. Tu jefe responderá que el precio de las casas es usualmente estimado manualmente por expertos: Un grupo recolecta información actual acerca del distrito, y cuando ellos no pueden obtener los media de el precio de las cosas, lo estiman usando reglas complejas. 

Esto es costoso en el consumo de tiempo, y sus estimaciones no son buenas: en algunos casos donde se desempeñan buscando el precio media de las casas, ellos realizan sus estimados estaban fuera por 20%. Esto es porque las compañía piensa en que sería útil entrenar un modelo para predecir el precio medio de las cosas por distrito, dando otros datos acerca del distrito. Los datos del censo parecen ser un excelente conjunto para explotar con este propósito, ya que incluye la media del precio de las casas por cientos de distritos, junto a otros datos. 

Con toda esa información, tu estar preparado para empezar a diseñar tu sistema. Primero, tu necesitas observar el problema: es supervised, unsupervised, o reinforcement learning? Es una tarea de clasificación, a regression, o algo más? Deberías usar batch learning o online learning? 

>[!tip] Si los datos son gigantescos, tu también podrías dividir tu batch learning work a través de multiples servidores (usando *map reduce*) o usando online learning

## Seleccionar un medida de rendimiento.

Tu siguiente paso es seleccionar una medida de rendimiento. Una típica medida de rendimiento para problemas de regresión es *Root Mean Square Error (RMSE)*. Provee una idea de cuanto error el sistema típicamente hace en sus predicciones, con un peso mayor para errores grandes. 

$$
\begin{equation}
\text{RMSE}(X, h) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( h(x^{(i)}) - y^{(i)} \right)^2 }
\end{equation}
$$
>[!Note] Notations
>Esta ecuación introduce algunas notaciones comunes en ML que se usaran a lo largo del libro:
>- *m* el el número de instancias en el dataset que se esta midiendo el RMSE
>	- For ejemplo, si tu estas evaluando el RMSE en un set de validación de 2.000 distritos, entonces m = 2000
>- *$x^{t}$* es un vector de todos los valores de las features (excluyendo la label) of the $i^{th}$ estancia en el dataset, y $y^{t}$ es la label (el output deseado por cada instancia).
>	- Por ejemplo, si el primer distrito en el dataset esta localizado en la longitud -119.29 latitud 33.91, y tiene 1416 habitantes con una media salarial de 38372, y el valor medio de una casa es 156400 (ignorando las otras features por ahora), entonces: 

$$
\begin{equation}
\mathbf{x}^{(1)} =
\begin{pmatrix}
-118.29 \\
33.91 \\
1,416 \\
38,372
\end{pmatrix}
\end{equation}
$$
$$
\begin{equation}
y^{(1)} = 156,400
\end{equation}
$$
>[!Note]
>- *X* es una matriz que esta conteniendo todos los Feature values (excluyendo las labels) de todas las instancias en el data set hay una fila por cada instancia, y la $i^{th}$ fila es igual a la transpuesta de $x^t$, notado $(x^t)^T$ 
>	-Por ejemplo, si el primer distrito es como se lo describió, entonces la matriz X se mirara así: 

$$
\mathbf{X} =
\begin{pmatrix}
(\mathbf{x}^{(1)})^T \\
(\mathbf{x}^{(2)})^T \\
\vdots \\
(\mathbf{x}^{(1999)})^T \\
(\mathbf{x}^{(2000)})^T
\end{pmatrix}
=
\begin{pmatrix}
-118.29 & 33.91 & 1,416 & 38,372 \\
\vdots & \vdots & \vdots & \vdots
\end{pmatrix}
$$

>[!Note]
>- *h* es la función de predicción de tu sistema, también llamado *hypothesis*. Cuando tu sistema se le da una instancia de feature vector $x^t$, eso resultara en un valor de predicción $\hat{y}^t=h(x^t)$ para esa instancia ($\hat{t}$ es llamada y-hat).
>	- Por ejemplo, si tu sistema predice que la media del valor de las casas en el primer distrito es 158400 entonces $\hat{y}^{1}=h(x^t)=158400$ El error de la predicción para este distrito es $\hat{y}^1- y^1=2000$
>- RMSE(X,h) es el costo de la función medido en el set de ejemplos usando la hypothesis h.


## Check the assumptions 

Últimamente, es una buena practica listar y verificar las asunciones que se tiene hechas hasta ahora (por ti o por otros); esto puede ayudarte a percatarte de serios peligros temprano, Por ejemplo, los precios por distrito que el sistema esta dando como output serán parte del alimento del rio abajo del ML system, y tu asumes que los precios van a ser usados como tal. Pero que si el downstream system convierte los precios en categorías. y entonces el uso de estas categorías en lugar el precio como tal? Es este caso, tener el precio perfectamente no es tan importante en absoluto; tu sistema solo necesita tener las categorías correctamente. Si ese es el caso, el problema tiene que estar enfocado en classification, no regression. Tu no quieres encontrar eso después de trabajar con un regression system por meses. 

Afortunadamente, después de hablarlo con el equipo a cargo del downstream system, tu estas en confianza que ellos necesitan los precios reales, no solo categorías. Genial! Tu estas preparado, las luces son verdes, y tu puedes empezar coding ahora! 


## Project 

### Percentiles 

Los percentiles ayudan a entender la distribución de los datos dividiéndolos en partes. En este caso, se muestran los valores correspondientes al **25% (primer cuartil)**, **50% (segundo cuartil o mediana)** y **75% (tercer cuartil)**.

- **25% (primer cuartil - Q1):** Un 25% de los datos son **iguales o menores** a este valor.
    - Por ejemplo, en la columna **median_house_value**, el valor de **119,600** indica que el 25% de las casas tienen un precio igual o menor a esa cantidad.
- **50% (segundo cuartil - Q2 o mediana):** Es el punto medio de los datos; la mitad de los valores están por debajo y la otra mitad por encima.
    - En **median_house_value**, el valor de **179,700** significa que el 50% de las casas tienen un precio menor o igual a ese valor.
- **75% (tercer cuartil - Q3):** El 75% de los datos están **por debajo** de este valor, mientras que el 25% restante están **por encima**.
    - Para **median_house_value**, el valor de **264,725** indica que tres cuartas partes de las casas tienen un precio menor o igual a ese número.

### Histogram

Un histogram muestra el número de instancias dentro de un rango específico. 
- El eje horizontal (X) se representan los valores de un atributo numérico dividido en rangos (o "bins").
- En el eje vertical (Y) se muestra la cantidad de veces que aparecen valores dentro de cada rango 
Ayuda a entender como están distribuidos los datos en un conjunto. 
- Si hay valores más frecuentes en cierta zona.
- Si hay distribuciones simétrica o sesgada hacia un lado.
- Si hay valores atípicos (outliers).
- 