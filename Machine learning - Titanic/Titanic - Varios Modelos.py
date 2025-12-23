import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. CARGA DE DATOS
# -------------------------------------------------------------------------
print("Cargando datos...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. DEFINICIÓN DE LOS 3 CONJUNTOS DE VARIABLES (ESCENARIOS)
# -------------------------------------------------------------------------

# Escenario 1: "Todas las variables" (Excepto IDs y textos complejos como Name/Ticket/Cabin)
# Nota: Usamos las variables estándar que aportan valor predictivo sin ingeniería compleja.
cols_todas = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Escenario 2: "Variables Específicas" (Edad, Sexo, Pclass, Fare)
cols_subset = ['Age', 'Sex', 'Pclass', 'Fare']

# Escenario 3: "Stepwise" usará 'cols_todas' pero el modelo seleccionará las mejores automáticamente.

# 3. PREPROCESAMIENTO
# -------------------------------------------------------------------------
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_cols = ['Pclass', 'Sex', 'Embarked']

# Pipeline numérico: Imputación + Escalado (Vital para SVM/KNN)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline categórico: Imputación + OneHot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocesador base
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 4. CONFIGURACIÓN DEL TORNEO
# -------------------------------------------------------------------------
# Lista de algoritmos a probar
algoritmos = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ("SVM", SVC(random_state=42)),
    ("KNN", KNeighborsClassifier())
]

# Definimos los 3 escenarios de datos
# Formato: (Nombre, Lista_Columnas, Usar_Stepwise)
escenarios = [
    ("Modelo 1: Todas las Variables", cols_todas, False),
    ("Modelo 2: Subset (Age, Sex, Pclass, Fare)", cols_subset, False),
    ("Modelo 3: Stepwise Selection (Automático)", cols_todas, True) 
]

# Variables para guardar al ganador
mejor_score_global = 0
mejor_modelo_global = ""
mejor_pipeline_global = None
mejor_escenario_global = ""

print("\n--- INICIANDO COMPARATIVA MASIVA (3 Escenarios x 6 Modelos) ---\n")

# 5. EJECUCIÓN DEL BUCLE (ESCENARIOS x MODELOS)
# -------------------------------------------------------------------------
for nombre_escenario, columnas, usar_stepwise in escenarios:
    print(f"\n>> Evaluando: {nombre_escenario}...")
    
    # Preparamos los datos X e y para este escenario
    # Nota: Filtramos las columnas que existen en 'columnas' 
    # (Para el subset, ignoramos cols que no estén en la lista)
    cols_activas_num = [c for c in numerical_cols if c in columnas]
    cols_activas_cat = [c for c in categorical_cols if c in columnas]
    
    # Redefinimos el preprocesador dinámicamente según las columnas del escenario
    preprocessor_actual = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, cols_activas_num),
            ('cat', categorical_transformer, cols_activas_cat)
        ])

    X = train_df[columnas]
    y = train_df['Survived']
    
    # Split Train/Val
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    for nombre_algo, modelo in algoritmos:
        pasos = [('preprocessor', preprocessor_actual)]
        
        # Si es el escenario Stepwise, añadimos el selector de características
        if usar_stepwise:
            # Usamos LogisticRegression como selector rápido para filtrar variables
            sfs = SequentialFeatureSelector(estimator=LogisticRegression(max_iter=1000), 
                                          n_features_to_select='auto', 
                                          direction='forward',
                                          tol=None,
                                          cv=3)
            pasos.append(('feature_selection', sfs))
        
        # Añadimos el modelo final
        pasos.append(('model', modelo))
        
        pipeline = Pipeline(steps=pasos)
        
        # Entrenar
        pipeline.fit(X_train, y_train)
        
        # Validar
        preds = pipeline.predict(X_valid)
        score = accuracy_score(y_valid, preds)
        
        print(f"   -> {nombre_algo: <25}: {score:.4f}")
        
        # Comprobar si es el nuevo campeón
        if score > mejor_score_global:
            mejor_score_global = score
            mejor_modelo_global = nombre_algo
            mejor_escenario_global = nombre_escenario
            mejor_pipeline_global = pipeline

# 6. RESULTADOS Y PREDICCIÓN FINAL
# -------------------------------------------------------------------------
print("\n" + "="*60)
print(f"🏆 GANADOR ABSOLUTO:")
print(f"Escenario: {mejor_escenario_global}")
print(f"Algoritmo: {mejor_modelo_global}")
print(f"Precisión: {mejor_score_global:.4f}")
print("="*60)

print(f"\nGenerando predicciones finales para 'test.csv' usando el ganador...")

# Preparamos el test set con las columnas correctas según el escenario ganador
# (Recuperamos las columnas del escenario ganador buscando en la lista original)
cols_ganadoras = next(cols for nombre, cols, _ in escenarios if nombre == mejor_escenario_global)
X_test = test_df[cols_ganadoras]

predicciones = mejor_pipeline_global.predict(X_test)

output = pd.DataFrame({
    'PassengerId': test_df.PassengerId,
    'Survived': predicciones
})

archivo_salida = 'submission_titanic_final.csv'
output.to_csv(archivo_salida, index=False)
print(f"✅ Archivo guardado: {archivo_salida}")