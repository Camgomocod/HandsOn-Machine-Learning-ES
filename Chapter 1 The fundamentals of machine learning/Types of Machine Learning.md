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

![[Pasted image 20250225102910.png]]

Un ejemplo típico de entrenamiento supervisado es classification. El modelo es entrenado con muchos ejemplos de corres a lo largo de su clase (spam or not), y este debe aprender a como clasificar estos correos. Otra tarea típica es target numerical value, como el precio de un carro, a partir de unas características dadas. (millas, año, marca etc) llamados predictors. Esa lista de tareas es llamada Regression. Para entrenar el sistema hay que darle muchos ejemplos de carros incluidos las labels y los predictors.

*Note* algunos regression algorithms pueden ser usados para classification, y vise versa. Por ejemplo modelos de Logistic Regression pueden ser usados para clasificar, según el valor de salida corresponda a la probabilidad de pertenecer a una clase dada (e,f 20% chance of spam)

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

![[Pasted image 20250225104554.png]]

Visualization algorithms son también buenos ejemplos de entrenamiento sin supervisión. Se los alimenta con muchos datos complicados sin label, y ellos tienen un 2d o 3d representación de los datos que fácilmente pueden ser graficados. Estos algoritmos intentan preservar la estructura tanto como puedan (tratando de separar por clusters en el espacio de salida para mapear una visualización) así podemos entender como los datos están siendo organizados, para talvez identificar patrones que no han tenido en cuenta. 

![[Pasted image 20250225104938.png]]

Una tarea relacionada es dimensionality reduction, el cual es el objetivo de simplificar los datos sin perder demasiada información. Una de las formas de hacer esto es unir unir algunas de las características en una. Por ejemplo el kilometraje esta fuertemente relacionado con el año, así que la reducción de dimensionalidad unirá usas dos una una sola feature (característica) que represente el desgaste del carro. Esto es llamado feature extraction. 

>[!Note] Usualmente es buna idea intentar reducir la dimensionalidad de los datos de entrenamiento usando dimensionality reduction algorithm antes de darlo como entrenamiento para otro ML (como supervised training)  eso hará que corra mucho más rápido, tome menos disco en memoria, y en algunos casos que rinda mejor. 

Otra de las tareas importantes en unsupervised learning es anomaly detection. Por ejemplo detectar transacciones fraudulentas de tarjetas de crédito para prevenirlos, detectando defectos de mano factura, o automáticamente removiendo outliers (valores atípicos) antes de usar ese dataset para entrenamiento. Por lo los data set se muestran normales por lo que hay que aprender a reconocerlos, para que después en nuevas instancias pueda detectarlo como normal o anomalía 

![[Pasted image 20250225111445.png]]

Otra tarea para unsupervised es en association rule learning, el propósito es excavar entre grandes cantidades de datos para descubrir relaciones interesantes entre atributos. Por ejemplo suponga un super mercado. Running an association rule en las ventas puede revelar que las personas que compran salsa barbecue y papas tienen la tendencia a comprar chuletas. Esto es interesante para colocar esos items cerca entre si. 

### Semi supervised learning 

Desde que etiquetar datos es usualmente costoso en tiempo y dinero, a menudo se tiene muchas instancias sin etiquetar, y pocas etiquetadas. Algunos algoritmos pueden lidiar con datos que están parcialmente etiquetados. Esto es llamado semi supervised learning. 

![[Pasted image 20250225112124.png]]

Algunos photo-hosting services, como google photos, son buenos ejemplos de esto. Una vez que se suba una foto familiar al servicio, este automáticamente reconoce que una persona esta en una foto C, H y otra persona en H y E. Esto es la parte unsupervised learning (clustering). Ahora el sistema necesita del usuario para darle un nombre a dichas personas. Agregando el nombre por persona el servicio ahora será capaz de nombrarlo en todas las fotos que aparezca, que es muy util para buscar fotos. 

### Reinforcement learning

Este caso es bastante diferente. El sistema de aprendizaje llamado agent en este contexto, puede observar su entorno, seleccionar y realizar acciones, y al agente tiene recompensas o penalidades de regreso. Este debe entender por el mismo cual es la mejor estrategia, llamada policy, para obtener la mayor cantidad de recompensas a lo largo del tiempo. A policy define que acción el agente debe tomar dada una situación. 

![[Pasted image 20250225113041.png]]

### Batch and Online Learning 

Otro criterio usado para clasificar Machine Learning systems es sí o no son capaces de aprender de forma incremental a partir de el flujo de datos entrantes. 

#### Batch learning 
(Aprendizaje por lotes) 

En Batch learning los sistemas son incapaces de aprender de forma incremental: Deben ser entrenado usando todos los datos disponibles. Esto generalmente toma mucho tiempo y poder computacional. Así que comúnmente es hecho offline. Primero el sistema es entrenado, luego es desplegado para producción y corre sin aprender nunca más; solo aplica lo que a aprendido. Esto es llamado offline learning.

Si se quiere batch learning sepa acerca los nuevos datos (como un nuevo tipo de spam) se necesita entrenar una nueva versión del sistema desde 0 sobre todo el dataset (la antigua y los nuevos datos) luego se para el sistema viejo para remplazarlo con el nuevo. 

Afortunadamente todo el proceso de entrenamiento, evaluación y despliegue de Machine Learning system puede ser automatizado bastante fácil. Así que incluso batch learning system puede adaptarse al cambio. Simplemente actualizando los datos y entrenar nuevas versiones del modelo. Esta solución es simple pero no siempre funciona bien ya que entrenar un modelo desde 0 con los los viejos u nuevos datos puede tomar horas (dependiendo de las dimensiones del dataset) así que se puede escoger entrenar el modelo cada día o semana. Si se necesita ser reactivo a los nuevos datos, se necesita una solución mas reactiva. 

También el entrenamiento requiere de muchos recursos computacionales. Si se tiene muchos datos y se automatiza el sistema de entrenamiento desde 0 todos los días, va a retornar un montón de costos. Si la proporción de datos es enorme puede ser imposible aplicar este tipo de algoritmo. 

Finalmente si el sistema necesita aprender con autonomía y tiene recursos limitados. El cargar grandes cantidades de datos y que tome horas el nuevo entrenamiento tomando grandes cantidades de recursos entrenando por horas puede ser terrible solución. (hay mejores opciones para este tipo de casos) algorithms que sean capases de aprender de forma incremental. 

#### Online learning

