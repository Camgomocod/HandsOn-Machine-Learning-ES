# Machine Learning Landscape

**Creación:** 2025-02-25 09:51
[[Chapter 1 The fundamentals of machine learning]]

Un ejemplo general que propone el libro lo hace sobre el tema de correos de SPAM, propone un ejemplo inicial de como clasificar estos correos usando una programación tradicional, determinando reglas y patrones. El proceso que sigue ese proceso es el siguiente.

![Figure 1-1. Enfoque tradicional](/assets/chapter1/Pasted%20image%2020250225095754.png)

![[Pasted image 20250225095754.png]]

Por el contrario un filtro basado en técnicas de Machine Learning aprende que palabras son buenas predicciones para tomar el correo como spam, detectando patrones de palabras y haciendo relaciones. Haciendo que el programa sea mucho más corto y fácil de mantener.  Si los spammers hacen diferentes patrones o cambian su redacción en un enfoque tradicional tendremos que escribir nuevas reglas para cada uno de ellos. 

![Figura 1-2, Figura 1-3](/assets/chapter1/Pasted%20image%2020250225100116.png)
![[Pasted image 20250225100116.png]]

Una vez se haya realizado iteraciones sobre el enfoque de entrenamiento de un modelo, se lo puede usar para entender mejor el problema, ya que se puede analizar los patrones, combinaciones que el modelo vea como las mejores predicciones por ejemplo para detectar un correo como spam. Muchas veces puede revelar patrones o entendimientos que pasaron sin sospecha en la primera iteración del entrenamiento. Aplicado técnicas de ML para  excavar dentro de grandes cantidades de datos, puede ayudar a descubrir patrones que inmediatamente no fueron tomados en cuenta. Eso es llamado Data Mining

![Figura 1-4](/assets/chapter1/Pasted%20image%2020250225100838.png)
![[Pasted image 20250225100838.png]]

- Simplifica el código y tiene un rendimiento mejor que un enfoque tradicional que requiere un montón de reglas que cambian con el tiempo, de lo cual no hay forma de automatizar. 
- Para los ambientes que son variables ML puede adaptarse a esos nuevos datos que se generen 
- Obtener información sobre problemas complejos y grandes cantidades de datos. 

## Examples 

Recommending a product that a client may be interested in, based on past purchases
This is a recommended system. One approach is to feed past purchases (and
other information about the client) to an artificial neural network

Segmenting clients based on their purchases so that you can design a different marketing
strategy for each segment.

Forecasting your company’s revenue next year, based on many performance metrics
This is a regression task (i.e., predicting values) that may be tackled using any
regression model, such as a Linear Regression or Polynomial Regression model



> Podemos hacer como practica de machine learning el extraer datos de las guías de mercado libre

