# Types of Machine Learning

**Creación:** 2025-02-25 10:19
[[Chapter 1 The fundamentals of machine learning]]

- Si están supervisados por humanos. (supervised, unsupervised, semi supervised, and Reinforcement learning)
- Si pueden o no aprender de forma incremental sobre la marcha (online versus batch (lote) learning)
- Si ellos trabajan simplemente comparando los datos con los nuevos puntos de datos para saber nuevos data points, o por otro lado detectando patrones en el entrenamiento y construyendo un modelo predictivo (instance-based versus model-based learning)

Estos criterios no son exclusivos, pueden ser combinados según sea el criterio/

## Supervised/Unsupervised Learning

Los sistemas de ML pueden ser clasificados según la cantidad y el tipo de supervisión que tengan durante el entrenamiento. Hay cuatro categorías: supervised learning, unsupervised learning, semi supervised learning, y Reinforcement learning. 

### Supervised learning 

En el entrenamiento supervisado lo que se hace es que con el data set que se da de alimento para el algoritmo incluye la solución deseada, llamada label. 

![Figura 1-5](/assets/chapter1/Pasted%20image%2020250225102910.png)
![[Pasted image 20250225102910.png]]

Un ejemplo típico de entrenamiento supervisado es classification. El modelo es entrenado con muchos ejemplos de correos a lo largo de su clase (spam or not), y este debe aprender a como clasificar estos correos. Otra tarea típica es target numerical value, como el precio de un carro, a partir de unas características dadas. (millas, año, marca etc) llamados predictors. Esa lista de tareas es llamada Regression. Para entrenar el sistema hay que darle muchos ejemplos de carros incluidos las labels y los predictors.

*Note* algunos regression algorithms pueden ser usados para classification, y vise versa. Por ejemplo modelos de Logistic Regression pueden ser usados para clasificar, según el valor de salida corresponda a la probabilidad de pertenecer a una clase dada (e,f 20% chance of spam)

![Figura 1-6](/assets/chapter1/Pasted%20image%2020250225103531.png)
![[Pasted image 20250225103531.png]]

Algunos de los algoritmos más importantes para supervised learning son los siguientes: 

- k-Nearest Neighbors
- Linear Regression 
- Logistic Regression 
- Support Vector Machines 
- Decision Trees and Random Forest
- Neural Networks^2

### Unsupervised Learning 

Para este tipo de modelo los datos están sin clasificar unlabeled. El sistema intenta aprender sin ningún profesor 

![Figura 1-7](/assets/chapter1/Pasted%20image%2020250225103819.png)
![[Pasted image 20250225103819.png]]

Estos son algunos de los algoritmos más importantes para unsupervised learning: 

- Clustering 
	- k Means 
	- DBSCAN
	- Hierarchical Cluster Analysis (HCA)
- Anomaly detection and novelty detection 
	- One-class SVM
	- Isolation Forest 
- Visualization and dimensionality reduction
	- Principal Component Analysis (PCA)
	- Kernel PCA
	- Locally Linear Embedding (LLE)
	- t-Distributed \Stochastic Neighbor Embedding (t-SNE)
- Association rule learning
	- Apriori
	- Eclat

Un ejemplo puede ser para analizar los visitas de un blog para hacer clustering de las diferentes clasificaciones que se pueden formar entre ellos, por ejemplo un 40% son mujeres que le gustan los comics y generalmente leen el blog en la mañana. mientras que 20% son hombres que le gustan historias románticas durante los fines de semana. Si se usa hierarchical clustering, se pude subdividir esos grupos en más pequeños, esto puedo ayudar a hacer Posts para cada grupo. 

![Figura 1-8](/assets/chapter1/Pasted%20image%2020250225104554.png)
![[Pasted image 20250225104554.png]]

Visualization algorithms son también buenos ejemplos de entrenamiento sin supervisión. Se los alimenta con muchos datos complicados sin label, y ellos tienen un 2d o 3d representación de los datos que fácilmente pueden ser graficados. Estos algoritmos intentan preservar la estructura tanto como puedan (tratando de separar por clusters en el espacio de salida para mapear una visualización) así podemos entender como los datos están siendo organizados, para talvez identificar patrones que no han tenido en cuenta. 

![Figura 1-9](/assets/chapter1/Pasted%20image%2020250225104938.png)
![[Pasted image 20250225104938.png]]

Una tarea relacionada es dimensionality reduction, el cual es el objetivo de simplificar los datos sin perder demasiada información. Una de las formas de hacer esto es unir unir algunas de las características en una. Por ejemplo el kilometraje esta fuertemente relacionado con el año, así que la reducción de dimensionalidad unirá usas dos una una sola feature (característica) que represente el desgaste del carro. Esto es llamado feature extraction. 

>[!Note] Usualmente es buna idea intentar reducir la dimensionalidad de los datos de entrenamiento usando dimensionality reduction algorithm antes de darlo como entrenamiento para otro ML (como supervised training)  eso hará que corra mucho más rápido, tome menos disco en memoria, y en algunos casos que rinda mejor. 

Otra de las tareas importantes en unsupervised learning es anomaly detection. Por ejemplo detectar transacciones fraudulentas de tarjetas de crédito para prevenirlos, detectando defectos de mano factura, o automáticamente removiendo outliers (valores atípicos) antes de usar ese dataset para entrenamiento. Por lo los data set se muestran normales por lo que hay que aprender a reconocerlos, para que después en nuevas instancias pueda detectarlo como normal o anomalía 

![figura 1-10](/assets/chapter1/Pasted%20image%2020250225111445.png)
![[Pasted image 20250225111445.png]]

Otra tarea para unsupervised es en association rule learning, el propósito es excavar entre grandes cantidades de datos para descubrir relaciones interesantes entre atributos. Por ejemplo suponga un super mercado. Running an association rule en las ventas puede revelar que las personas que compran salsa barbecue y papas tienen la tendencia a comprar chuletas. Esto es interesante para colocar esos items cerca entre si. 

### Semi supervised learning 

Desde que etiquetar datos es usualmente costoso en tiempo y dinero, a menudo se tiene muchas instancias sin etiquetar, y pocas etiquetadas. Algunos algoritmos pueden lidiar con datos que están parcialmente etiquetados. Esto es llamado semi supervised learning. 

![Figura 1-11](/assets/chapter1/Pasted%20image%2020250225112124.png)
![[Pasted image 20250225112124.png]]

Algunos photo-hosting services, como google photos, son buenos ejemplos de esto. Una vez que se suba una foto familiar al servicio, este automáticamente reconoce que una persona esta en una foto C, H y otra persona en H y E. Esto es la parte unsupervised learning (clustering). Ahora el sistema necesita del usuario para darle un nombre a dichas personas. Agregando el nombre por persona el servicio ahora será capaz de nombrarlo en todas las fotos que aparezca, que es muy util para buscar fotos. 

### Reinforcement learning

Este caso es bastante diferente. El sistema de aprendizaje llamado agent en este contexto, puede observar su entorno, seleccionar y realizar acciones, y al agente tiene recompensas o penalidades de regreso. Este debe entender por el mismo cual es la mejor estrategia, llamada policy, para obtener la mayor cantidad de recompensas a lo largo del tiempo. A policy define que acción el agente debe tomar dada una situación. 

![Figura 1-12](/assets/chapter1/Pasted%20image%2020250225113041.png)
![[Pasted image 20250225113041.png]]

## Batch and Online Learning 

Otro criterio usado para clasificar Machine Learning systems es sí o no son capaces de aprender de forma incremental a partir de el flujo de datos entrantes. 

### Batch learning 
(Aprendizaje por lotes) 

En Batch learning los sistemas son incapaces de aprender de forma incremental: Deben ser entrenado usando todos los datos disponibles. Esto generalmente toma mucho tiempo y poder computacional. Así que comúnmente es hecho offline. Primero el sistema es entrenado, luego es desplegado para producción y corre sin aprender nunca más; solo aplica lo que a aprendido. Esto es llamado offline learning.

Si se quiere batch learning sepa acerca los nuevos datos (como un nuevo tipo de spam) se necesita entrenar una nueva versión del sistema desde 0 sobre todo el dataset (la antigua y los nuevos datos) para así el sistema viejo detenerse y remplazarlo con el nuevo. 

Afortunadamente todo el proceso de entrenamiento, evaluación y despliegue de Machine Learning system puede ser automatizado bastante fácil. Así que incluso batch learning system puede adaptarse al cambio. Simplemente actualizando los datos y entrenar nuevas versiones del modelo. Esta solución es simple pero no siempre funciona bien ya que entrenar un modelo desde 0 con los viejos o nuevos datos puede tomar horas (dependiendo de las dimensiones del dataset) así que se puede escoger entrenar el modelo cada día o semana. Si se necesita ser reactivo a los nuevos datos, se necesita una solución mas reactiva. 

También el entrenamiento requiere de muchos recursos computacionales. Si se tiene muchos datos y se automatiza el sistema de entrenamiento desde 0 todos los días, va a retornar un montón de costos. Si la proporción de datos es enorme puede ser imposible aplicar este tipo de enfoque. 

Finalmente si el sistema necesita aprender con autonomía y tiene recursos limitados. El cargar grandes cantidades de datos y que tome horas el nuevo entrenamiento tomando grandes cantidades de recursos entrenando por horas puede ser terrible solución. (hay mejores opciones para este tipo de casos) algorithms que sean capases de aprender de forma incremental. 

### Online learning

El sistema es entrenado de manera incremental secuencialmente, o en pequeños grupos llamados mini-batches. Cada paso de aprendizaje es rápido y barato, así que el sistema puede aprender acerca los nuevos datos en el proceso de llagada. 

![Figura 1-13](/assets/chapter1/Pasted image 20250228152048.png)
![[Pasted image 20250228152048.png]]

Online learning es muy bueno para los sistemas que reciben flujos de datos continuamente (stock prices) y necesita adaptarse a los cambios rápidamente o automáticamente. También es una buena opción cuando se tiene recursos computacionales limitados. Una vez que el sistema halla aprendido de las nuevas instancias de datos, estás no son necesitadas nunca más, a excepción de un roll-back a un cierto estado. De esa forma se puede ahorrar un montón de espacio. 

Los learning algorithms pueden ser usados también para entrenar sistemas con datasets enormes que no pueden encajar en la memoria de la maquina, (esto es llamado out-of-core learning). El algoritmo carga parte de los datos, se entrena con esa parte, y se repite el proceso hasta que se haya entrenado con todos los datos. 

>[!Note] Out-of-core learning es usualmente realizado off-line. Se puede ver como un enfoque de aprendizaje incremental. 

Un parámetro importante a tener en cuneta es el como de rápido los online systems pueden adaptarse a los nuevos datos llamado learning rate. Si se configura un ratio alto el sistema se adaptara rápidamente, pero esto genera una tendencia a olvidar los datos anteriores. Por en contrario si se configura un ratio de aprendizaje bajo, el sistema puede tener mayor inercia, eso es que va a aprender lentamente, y también va a ser menos sensible al ruido de los nuevos datos entrantes, o de malos data points que se desborden del comportamiento de los datos. 

![Figure 1-14](/assets/capter1/Pasted image 20250228153302.png)
![[Pasted image 20250228153302.png]]

Uno de los mayores retos es que si el sistema es alimentado con malos datos. El rendimiento disminuirá gradualmente. Si es un sistema que corren en vivo, los clientes lo notarán. Si se detecta que el sistema tiene caídas de rendimiento. Se necesita monitor los datos entrantes por si se comportan de manera anormal. (usando un anomaly detection algorithm).

## Instance-Based Versus Model-Based Learning

Una forma más de categorizar sistemas ML es por como estos generalizan. Muchas de las tareas de ML es hacer predicciones. Esto quiero decir que dado un número de ejemplos de entrenamiento, el sistema necesita ser bueno para hacer predicciones (generalizar) con ejemplos que nunca a visto antes. Teniendo una buena medida de rendimiento con relación a los datos de entrenamiento es bueno, pero insuficiente; el verdadero objetivo es tener un buen rendimiento en nuevas instancias. 

Hay dos principales enfoques de generalización: instance-based learning and model-based learning. 

### Instance-based learning

Posiblemente la mas trivial forma de aprendizaje, es simplemente hacerlo por memoria. Si se crea un filtro de spam de esta forma, solo tomaría que se señalar todos los emails que sean idénticos a los señalados por los usuarios- no es la peor solución pero definitivamente no la mejor. 

En lugar de señalar a los que sean directamente idénticos se puede tener una cierta mesura programando para señalar a los que también tengan un parentesco. Esto requiere a measure of similarity entre los dos correos.  (como contar el número de palabras que tienen en común).

Esto es llamado instance based learning el sistema aprende por memoria, nuevo generaliza para nuevos casos usando un sistema de semejanza para comparar con los ejemplos ya aprendidos. 

![[Pasted image 20250301151023.png]]

### Model based learning

Otra forma de generalizar una pila de ejemplos es construir un modelo de esos ejemplos y luego usarlo para hacer predicciones. 

![[Pasted image 20250301151134.png]]

Después de usar el modelo, se necesita tener parámetros para el bias (desvío) y el peso, por ejemplo al usar una regresión lineal que tiene la forma de y(x) = mx + b, donde x serían los parámetros de entrada con los cuales hacer las predicciones de interés. Se necesita especificar una medida de rendimiento para especificarlo. Se puede definir una fitness function que mide como de bueno es el modelo, o se puede definir una función de costo (cost function) esto mide como de mala es. Para problemas de regresión lineal se usa la distancia entre el valor predicho y un valor teórico del data set.  

En resumen: 

- Se estudia los datos 
- Se selecciona el modelo 
- Lo entrenamos con los datos de entrenamiento (i,e., los algoritmos de aprendizaje realizan una búsqueda de un parámetro del modelo para minimizan la función de costo)
- Finalmente aplicamos el modelo para hacer predicciones de nuevos casos (Esto es llamado inference (inferencia)), esperando que el modelo generalize correctamente. 

## Mayores retos de ML

En resumen la tarea principal es escoger un buen algoritmo de aprendizaje y entrenarlo con algunos datos, hay dos cosas que pueden salir mal, escoger un mal algoritmo y tener malos datos, empecemos con ejemplos de malos datos. 

### Cantidad insuficiente de datos de entrenamiento 

Para que una niña pequeña aprenda lo que una manzana toma, señalarle lo que es una manzana unas cuentas veces y esta será capaz de distinguirlas sin importar el color de las mismas, los ML algorithms no llegan aún a ese punto. Incluso para simples problemas al modelo hay que entrenarlo con grandes cantidades de datos para que aprenda de forma correcta. Y crece el número de datos necesarios con la complejidad del problema, como el reconocimiento de voz. 

![[Pasted image 20250301152835.png]]

En resumen dice que los algoritmos incluso los más simples llegan a producir los mismos resultados según sean alimentados con más y buenos datos. Considerando donde colocar el dinero en desarrollar un algoritmo más intrincado de datos o tener un dataset robusto.

### Non representative Training data 

En orden para generalizar de forma correcta es crucial que los datos de entrenamiento sean representativos de los nuevos casos que se quieren generalizar. Esto es es cierto en un sistema ya sea basado en instancia o en modelo. 

Por ejemplo la lista de países que se uso para el entrenamiento previo pueden ser no del todo representativos, algunos países fueron excluidos, lo siguiente muestra como se vería el modelo de regresión lineal si no se omite esos países. 

![[Pasted image 20250301154000.png]]

Si se entrena un linar model con esos datos se obtendrá una línea solida, mientras que el modelo antiguo es representado por la línea punteada. Como se puede notar el añadir algunos países puede afectar significativamente el comportamiento del modelo, pero esto deja claro que un modelo de regresión lineal nunca va a funcionar bien, parece que los países ricos no lo son tanto como los que lo son los moderados. (de hecho parecen ser infelices), y en cambio algunos países pobres parecen mas felices que muchos de los ricos. 

Es clave que el conjunto de entrenamiento sea representativo de los casos que se quiera generalizar. Esto es más complicado de lo que suena: si la muestra es muy pequeña se tendrá *sampling noise* (ruido de muestreo) (datos no representativos resultado del azar), pero incluso las muestras más grandes pueden ser no representativas si el método de muestreo es defectuoso. Esto es llamado *sampling bias* (sesgo de muestra).

![[Pasted image 20250301154811.png]]

### Poor-Quality Data

Obviamente si los datos de entrenamiento esta lleno de errores, *outliers* (valores atípicos) (debido a mediciones de mala calidad), esto hará difícil que el sistema detecto patrones subyacentes, así que el sistema tiene menos tendencia a rendir bien. A menudo es mejor gastar más tiempo limpiando los datos de entrenamiento. La verdad es que la mayoría de científicos de datos gastan la mayor parte de su tiempo haciendo solo eso. Lo siguientes son un par de ejemplos de cuando se quiere hacer una limpieza en los datos. 

- Si alguna de las instancias son claramente valores atípicos, puede ser de ayuda simplemente descartarlos o intentar resolver los errores manualmente. 
- Si alguna de las instancias están ignorando ciertas características (e,g, 5% de tus compradores no especificaron su edad), tu puedes decidir si o no ignorar ese atributo en total, ignorar esas instancias, llenar los datos faltantes (con datos de la media) o entrenar un modelo con ese atributo y otro sin el. 

### Irrelevant features

Tu sistema solo será capaz de aprender si los datos de entrenamiento contienen suficientes atributos relevantes y no muchos irrelevantes. Una parte critica del éxito de un proyecto ML es crear un buen conjunto de características con las que entrenar. Este proceso llamado *feature engineering* (ingeniería de características) implica seguir los siguientes pasos: 

- *Feature selection* (selección de características) seleccionar las características más útiles para entrenar el modelo a partir de las existentes. 
- *Feature extraction* (extracción de características) combinar características o atributos existentes para producir unas más útiles. Como se vio antes con la reducción de dimensionalidad. 

### Over fitting training data

Esto quiere decir el el modelo funciona bien con los datos de entrenamiento, pero no logra generalizar correctamente. El siguiente ejemplo muestra un gran sobreajuste sobre los datos de entrenamiento. Este funciona mejor sobre los datos de entrenamiento que uno de regresión lineal, pero no se puede confiar en las predicciones.

![[Pasted image 20250301175347.png]]

Modelos complejos como redes neuronales profundas pueden detectar patrones subyacentes en los datos, pero si el conjunto de datos tiene ruido, o es muy pequeño (lo que introduce ruido de muestreo), entonces el modelo tiene la tendencia en detectar patrones en el ruido como tal, obviamente esos patrones no se generalizarán en nuevas instancias de entrenamiento. Por ejemplo digamos que se alimenta nuestro modelo de satisfacción de vida con muchos mas atributos, incluyendo algunos que no tienen información relevante para el modelo como el nombre del país. un modelo complejo puede detectar patrones como el hecho de que todos los países del data set que contienen la letra w en sus nombres tienen una satisfacción mayor a 7. Cuanta es la confianza que esa regla aplique a países como Rwanda o Zimbabwe.  Obviamente este apron ocurre en el data set por pura casualidad, pero el modelo no tiene como distinguir si el patron es real o simplemente ruido. 

>[!Note] Overfitting (sobre ajuste) ocurre cuando el modelo es demasiado complejo con relación a los ruidos que tiene los datos de entrenamiento. Estas son posibles soluciones: 
>- Simplificar el modelo seleccionando alguno que tenga menos parámetros (linear model más que un high-degree polynomial model) reduciendo el número de atributos en los datos de entrenamiento o restringiendo el modelo
>- Recolectar más datos de entrenamiento 
>- Reducir el ruido en los datos de entrenamiento (arreglando los errores en los datos y removiendo las desviadores *outliers* (valores atípicos))

Restringiendo el modelo a uno más simple puede reducir el riego de sobre ajuste is llamado *regularization*. Por ejemplo en un modelo de regresión lineal se tiene dos atributos el bias y height, la pendiente y el intercepto, se puede restringir alguno de los dos parámetros para trabajar con alguno de los dos, o se puede hacer que uno de los dos atributos como la pendiente crezca con mesura para regularizar el modelo. 

En el siguiente ejemplo se puede ver un modelo con regularización, forzando al modelo a tener una menor pendiente, este modelo no se ajusta a los datos de entrenamiento (círculos) tan bien como el primer modelo, pero en realidad hace una mejor generalización con los nuevos ejemplos que no se ven durante el entrenamiento (cuadrados). 

![[Pasted image 20250301181004.png]]

La cantidad de regularización que se aplique durante el entrenamiento puede ser controlada por un *hyper parameter*. Esto es un parámetro del aprendizaje del algoritmo (no del modelo). Como tal, no se ve afectado por el aprendizaje del algoritmo como tal, debe ser establecido antes del entrenamiento y permanece constante durante. Si se configura el hiper parámetro a un valor muy grande, obtendrá un modelo casi plano (una pendiente cercana a cero); Es casi seguro que el modelo no se ajustará demasiado a los datos de entrenamiento, será menos probable que encuentre una buena solución. Ajustar los hiper parámetros es una parte importante durante la construcción de un sistema de aprendizaje automático.

### Underfitting the Training data 

Underfitting (desajuste) es el contrario a overfitting;  esto ocurre cuando tu modelo es muy simple para aprender la estructura subyacente de los datos. Por ejemplo un modelo de regresión lineal tiene la tendencia a un desajuste; la realidad es mucho más complejo que el modelo, así que las predicciones que realizará serán inexactas. Incluso con los datos de entrenamiento. 

Aquí están algunas de las opciones principales para arreglar este problema: 

- Seleccionar un modelo más poderoso 
- Alimentar con mejores características al algoritmo de aprendizaje (*feature engineering*)
- Reduciendo las restricciones del modelo ( reducir la regularización (*hyper-parameter*))

## Testing and validating 

La única manera en la que se puede saber que tan bien un modelo generalizará con los nuevos casos en realidad es intentar con nuevos casos. Una forma de hacer eso es poner el modelo en producción y monitorear que tan bien rinde. Esto funciona bien, si tu modelo es horriblemente malo, tus usuarios se quejaran - no es la mejor idea. 

La mejor opción es dividir los datos en dos lotes: Un *training set* (datos de entrenamiento) y un *test set* (lote de prueba), El error en los nuevos casos es llamado *generalization error* (error de generalización) , y evaluando tu modelo en los datos de prueba, puedes estimar ese error. Este valor te dice que tan bien tu modelo rinde en instancias que nunca a visto antes. 

Si el error de entrenamiento es bajo (tu modelo comete pequeños errores sobre los datos de entrenamiento) pero la generalización es alta, esto quiere decir que tu modelo esta sobre ajustado con los datos de entrenamiento.

>[!tip] Es común un uso del 80% de los datos para entrenamiento y mantener un 20% para los tests. Sin embargo esto depende del tamaño del dataset: Si este contiene 10 millones de instancias, el mantener 1% de los datos contendrá 100.000 instancias, probablemente más que suficiente para una buena estimación del error de generalización

### Hyperparameter tuning and model selection

Evaluar un modelo es bastante simple: solo usa el set de prueba, Pero supongamos que estas dudando entre dos tipos de modelos (linear regression y polinomial regression): como se puede decidir entre ellos? Una de las opciones es entrenar los dos para comparar como de bien generalizan usando el test set. 

Ahora supone que el modelo de regresión lineal generaliza mejor, pero quieres aplicar algunas regularizaciones para prevenir el sobreajuste. La pregunta es como se escoge el valor de la regularization hyperparameter? Una de las opciones es entrenar 100 modelos diferentes usando 100 valores diferentes para el hyperparameter. Suponga que encuentra el mejor de ellos que produce un modelo con el menor error en la generalización 5%. Despliegas este modelo a producción, pero infortunadamente no rinde tan bien como se espera, produciendo 15% de errores. Que paso? 

El problema es que se midió el error de generalización multiples veces en el conjunto de prueba, y se adapto el modelo y los hyperparameters para producir el mejor modelo para un conjunto de datos particular. Esto quiere decir que el modelo no tendrá tan buen rendimiento con datos nuevos. 

Una solución común a esto es llamada *holdout validation* (validación de retención): simplemente se aparta parte del set de entrenamiento para evaluar varios modelos candidatos y seleccionar el mejor. El nuevo held-out set is llamado *validation set* (conjunto de validación) (o algunos veces el set de desarrollo) Específicamente se entrena varios modelos con varios hyperparameters en el set de entrenamiento reducido. (todo el set de entrenamiento menos el set de validación), y se selecciona el modelo que mejor rinda en el conjunto de validación. Después de esta retención de el proceso de validación, se entrena el modelo con todo el set de entrenamiento (incluyendo el set de validación), y esto te da el modelo final. Por ultimo, tu evalúas el modelo final en el set de prueba para estimar el error de generalización. 

Esta solución suele funcionar bastante bien. Sin embargo, si el set de validación es muy pequeño, entonces las evaluaciones del modelo serán imprecisas, tu podrías terminar escogiendo un sub optimo modelo por error. En cambio, si el set de validación es muy grande, el set entrenamiento será mucho más pequeño que el set de entrenamiento completo. Porqué esto es malo? Bien, desde que el modelo final será entrenado con el tamaño completo del set. Eso sería como escoger a un velocista para un maratón. Una forma de resolver este problema es realizar repeticiones *cross-validation* (validación cruzada), usando muchos pequeños sets de validación. Cada modelo es evaluado por cada set de validación después es entrenado con el resto de los datos. Al promediar todas esas evaluaciones del modelo, tu puedes obtener una cifra mucho más precisa de su rendimiento. Sin embargo, hay un inconveniente. El tiempo de entrenamiento es multiplicado por el número de sets de validación. 

### Data mismatch 

En algunos casos es fácil conseguir grandes cantidades de datos de entrenamiento, pero estos datos probablemente no serán perfectamente representativas de los datos que serán usados en producción. Por ejemplo, suponga que quieres crear una app para hacer fotos de flores y que automáticamente determine cual especie es. Tu fácilmente puedes descargar millones de fotos de la web, pero esas no será perfectamente representativas de las fotos que en verdad van a ser tomadas usando un dispositivo móvil. Talvez tu solo tendrás 10,000 fotos representativas (las que son tomadas por la app). En este caso la regla más importante es que el set de validación y el de entrenamiento deben ser lo más representativos como sea posible de los datos que se espera en el uso de producción, así que estos deben estar compuestos exclusivamente de datos representativos. Se los puede mezclar colocando la mitad en el set de entrenamiento y otra parte en el de validación (estando seguro que duplicados (o muy parecidos) terminen en ambos sets) pero después de entrenar tu modelo con las fotos de la web, si observas que el rendimiento del modelo en el set de validación es decepcionante. Tu no sabrás si esto se debe a que tu modelo se ha sobre ajustado al conjunto de entrenamiento.
o si esto es solamente un desajuste entre los fotos de la web y las fotos tomadas desde la app móvil. Una solución es mantener algo de las fotos de entrenamiento (de la web) en otro conjunto que Andre Ng llama el set *train-dev* después que el modelo es entrenado (en el set de entrenamiento, no en train-dev set) , puedes evaluarlo con el train-dev set. Si este rinde bien, el modelo no esta sobre ajustado (overfitting) para los datos de entrenamiento. Si este rindo pobremente en el set de validación, el problema puede venir de el desajuste de los datos (data mismatch).  Puedes tratar de abordar este problema procesando las imágenes de la web para que parezcan más a las que serían tomadas por la app, y luego re entrenar el modelo. En cambio, si el model rinde pobremente en el train-dev set, este debería estar sobre ajustado en el set de entrenamiento, así que deberías intentar en simplificar o regularizar el modelo, obtener más datos de entrenamiento, y limpiar los datos de entrenamiento. 

![[Pasted image 20250303112533.png]]

# Exercises

In this chapter we have covered some of the most important concepts in Machine
Learning. In the next chapters we will dive deeper and write more code, but before we
do, make sure you know how to answer the following questions:
1. How would you define Machine Learning?
2. Can you name four types of problems where it shines?
3. What is a labeled training set?
4. What are the two most common supervised tasks?
5. Can you name four common unsupervised tasks?
6. What type of Machine Learning algorithm would you use to allow a robot to
walk in various unknown terrains?
7. What type of algorithm would you use to segment your customers into multiple
groups?
8. Would you frame the problem of spam detection as a supervised learning problem
or an unsupervised learning problem?
9. What is an online learning system?
10. What is out-of-core learning?
11. What type of learning algorithm relies on a similarity measure to make predictions?
12. What is the difference between a model parameter and a learning algorithm’s
hyperparameter?
13. What do model-based learning algorithms search for? What is the most common
strategy they use to succeed? How do they make predictions?
14. Can you name four of the main challenges in Machine Learning?
15. If your model performs great on the training data but generalizes poorly to new
instances, what is happening? Can you name three possible solutions?
17. What is a test set, and why would you want to use it?
18. What is the purpose of a validation set?
19. What is the train-dev set, when do you need it, and how do you use it?