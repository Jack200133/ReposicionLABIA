```python
import pandas as pd
import numpy as np

file = 'framingham.csv'

df = pd.read_csv(file)

data = df.to_numpy()

print(data)
```

    [[  1.  39.   4. ...  80.  77.   0.]
     [  0.  46.   2. ...  95.  76.   0.]
     [  1.  48.   1. ...  75.  70.   0.]
     ...
     [  0.  48.   2. ...  84.  86.   0.]
     [  0.  44.   1. ...  86.  nan   0.]
     [  0.  52.   2. ...  80. 107.   0.]]
    


```python
# TASK 1.2 
X = data[:, :-1]  # Todas las columnas excepto la última
y = data[:, -1]  # La última columna

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Crear un imputador para llenar los valores faltantes con la media de la columna
imputer = SimpleImputer(strategy='mean')

# Aplique el imputador a los datos
X_imputed = imputer.fit_transform(X)

# Continúe con el escalado y la ingeniería de características polinomiales como antes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

poly = PolynomialFeatures(degree=2)  # Ajuste el grado según lo desee
X_poly = poly.fit_transform(X_scaled)

# Divida los datos en conjuntos de entrenamiento y prueba, ajuste y evalúe el modelo como antes

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=123)

from sklearn.linear_model import LogisticRegression

# Ajustar el modelo logístico
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Realizar predicciones en el conjunto de prueba
y_pred = log_reg.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Imprimir un informe de clasificación
report = classification_report(y_test, y_pred)
print(report)



```

    Accuracy: 0.8301886792452831
                  precision    recall  f1-score   support
    
             0.0       0.84      0.98      0.91       708
             1.0       0.43      0.09      0.15       140
    
        accuracy                           0.83       848
       macro avg       0.64      0.53      0.53       848
    weighted avg       0.78      0.83      0.78       848
    
    


```python
# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de costo
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost

# Descenso del gradiente
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        theta = theta - (alpha/m) * (X.T @ (sigmoid(X @ theta) - y))
        J_history[i] = cost_function(X, y, theta)
    
    return theta, J_history

# Parámetros iniciales
m, n = X_train.shape
theta_initial = np.zeros(n)
alpha = 0.01
num_iters = 10000

# Ajustar el modelo
theta, J_history = gradient_descent(X_train, y_train, theta_initial, alpha, num_iters)

# Predecir usando el conjunto de prueba
y_test_prob = sigmoid(X_test @ theta)
y_test_pred = np.round(y_test_prob)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")

# Imprimir un informe de clasificación
report = classification_report(y_test, y_test_pred)
print(report)

```

    Accuracy: 0.8325471698113207
                  precision    recall  f1-score   support
    
             0.0       0.85      0.98      0.91       708
             1.0       0.47      0.10      0.16       140
    
        accuracy                           0.83       848
       macro avg       0.66      0.54      0.54       848
    weighted avg       0.78      0.83      0.78       848
    
    


```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('classifier', LogisticRegression(solver='saga', tol=1e-3, max_iter=10))
])

# Definir los grados de polinomio que desea evaluar
polynomial_degrees = [1, 2, 3,4,5,6]  # Reducir el rango de grados polinomiales

# Configurar la búsqueda en cuadrícula con validación cruzada
param_grid = {'poly__degree': polynomial_degrees}
grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=-1)

#80 train 10 test 10 valid
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=123)
X_test , X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

# Ajustar la búsqueda en cuadrícula
grid_search.fit(X_valid, y_valid)

# Obtener el mejor grado y el mejor modelo
best_degree = grid_search.best_params_['poly__degree']
best_model = grid_search.best_estimator_

print(f"El mejor grado polinomial es: {best_degree}")

# Ajustar el mejor modelo
best_model.fit(X_train, y_train)

# Predecir usando el conjunto de prueba
y_pred = best_model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


```

    El mejor grado polinomial es: 2
    Accuracy: 0.8278301886792453
    

## Analisis 
Basándonos en los resultados obtenidos para el modelo de regresión logística polinomial y el modelo de regresión logística, podemos hacer las siguientes observaciones:

1. Precisión y recall:

* Para la clase 0 (no sufre paro cardíaco), ambos modelos tienen una precisión y un recall altos. Esto indica que los modelos pueden predecir correctamente la mayoría de los casos en los que un paciente no sufre un paro cardíaco.
* Para la clase 1 (sufre paro cardíaco), ambos modelos tienen una precisión baja y un recall aún más bajo. Esto indica que los modelos tienen dificultades para predecir correctamente los casos en los que un paciente sufre un paro cardíaco. La mayoría de las predicciones de paro cardíaco son incorrectas (baja precisión) y muchos casos reales de paro cardíaco no se detectan (bajo recall).

2. Ajuste del modelo y complejidad:

* El mejor grado polinomial encontrado mediante la validación cruzada es 2. Esto sugiere que un modelo cuadrático puede ser suficiente para capturar la relación entre las variables independientes y la variable dependiente.
* La precisión general (accuracy) de ambos modelos es bastante similar, siendo ligeramente mayor en el modelo de regresión logística. Esto sugiere que el modelo polinomial de grado 2 no proporciona un aumento significativo en el rendimiento en comparación con el modelo de regresión logística lineal.

3. Sobre los resultados de cross-validation:

* El mejor grado polinomial encontrado fue 2, lo que sugiere que un modelo cuadrático puede ser suficiente para capturar la relación entre las variables independientes y la variable dependiente.
* La precisión en el mejor modelo polinomial es similar a la precisión de los modelos de regresión logística y logística polinomial sin validación cruzada. Esto indica que la validación cruzada proporciona una estimación razonable del rendimiento del modelo.  


En resumen, aunque se encontró que el mejor grado polinomial es 2, el modelo de regresión logística polinomial no ofrece una mejora significativa en la precisión en comparación con el modelo de regresión logística lineal. Ambos modelos tienen un rendimiento similar en términos de precisión y recall en cada clase. Sin embargo, en ambos casos, el rendimiento en la predicción de paros cardíacos (clase 1) es bajo y puede ser necesario explorar otras técnicas o características adicionales para mejorar el rendimiento del modelo en esta clase.
