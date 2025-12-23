# 🚢 Predicción de Supervivencia en el Titanic

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg" alt="Titanic" width="600">
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.9-blue" alt="Python">
    <img src="https://img.shields.io/badge/Library-Scikit--Learn-orange" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/Status-Finalizado-green" alt="Status">
</p>

## 📋 Descripción del Proyecto
Este proyecto utiliza técnicas de **Machine Learning** para predecir la supervivencia de los pasajeros del Titanic basándose en características demográficas y de viaje (clase, sexo, edad, tarifa, etc.).

El objetivo es desarrollar un modelo de clasificación binaria capaz de generalizar patrones de supervivencia con alta precisión.

## ⚙️ Tecnologías y Herramientas
* **Lenguaje:** Python
* **Librerías:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
* **Entorno:** VS Code.

## 📊 Análisis Exploratorio de Datos (EDA)
Se realizó un análisis Descriptivo para entender la distribución de variables y valores faltantes.

## 🛠️ Metodología

### 1. Preprocesamiento
* **Imputación de datos:** Se rellenaron valores nulos en `Age` utilizando la mediana agrupada por `Pclass` y `Sex`.
* **Feature Engineering:** Creación de la variable `FamilySize` (SibSp + Parch).
* **Codificación:** Uso de *One-Hot Encoding* para variables categóricas (`Embarked`, `Sex`).

### 2. Modelado
Se evaluaron los siguientes algoritmos:
1.  Regresión Logística
2.  Random Forest Classifier
3.  Support Vector Machines (SVM)

```python
# Ejemplo de configuración del modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)

Enlace: https://www.kaggle.com/competitions/titanic/overview
