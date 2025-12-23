import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estilo para gráficos
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def cargar_datos():
    """Carga los datasets y maneja errores básicos."""
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        print("❌ Error: Faltan los archivos 'train.csv' o 'test.csv'.")
        return None, None
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print("✅ Datos cargados correctamente.\n")
    return train, test

def analisis_estructura(df, nombre):
    """Imprime información básica sobre la estructura del DataFrame."""
    print(f"--- ESTRUCTURA DEL DATASET: {nombre} ---")
    print(f"Dimensiones: {df.shape}")
    print("\nTipos de datos y valores no nulos:")
    print(df.info())
    print("\nValores faltantes (nulos) por columna:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print("-" * 50 + "\n")

def estadisticas_descriptivas(df, nombre):
    """Muestra estadísticas de variables numéricas y categóricas."""
    print(f"--- ESTADÍSTICAS DESCRIPTIVAS: {nombre} ---")
    
    print("\n>> Variables Numéricas (Distribución):")
    print(df.describe().T)
    
    print("\n>> Variables Categóricas (Cardinalidad y Top):")
    # Seleccionamos solo columnas tipo object (texto/categoría)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(df[cat_cols].describe().T)
    else:
        print("No hay variables categóricas.")
    print("-" * 50 + "\n")

def plot_distribuciones_numericas(df, nombre):
    """Genera histogramas para variables numéricas."""
    nums = df.select_dtypes(include=['float64', 'int64']).columns
    # Excluimos IDs si existen
    nums = [c for c in nums if 'Id' not in c and 'Survived' not in c]
    
    if len(nums) > 0:
        df[nums].hist(bins=20, figsize=(14, 10), layout=(3, 3), color='skyblue', edgecolor='black')
        plt.suptitle(f'Distribución de Variables Numéricas - {nombre}', fontsize=16)
        plt.tight_layout()
        filename = f"distribucion_numerica_{nombre}.png"
        plt.savefig(filename)
        print(f"📊 Gráfico guardado: {filename}")
        plt.close()

def plot_categoricas(df, nombre):
    """Genera gráficos de barras para variables categóricas clave."""
    cols_clave = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_clave):
        if col in df.columns:
            sns.countplot(x=col, data=df, ax=axes[i], palette='viridis')
            axes[i].set_title(f'Conteo de {col} ({nombre})')
            axes[i].set_ylabel('Cantidad')
        else:
            axes[i].set_visible(False)
            
    # Ocultar el último subplot si sobra
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    filename = f"conteo_categoricas_{nombre}.png"
    plt.savefig(filename)
    print(f"📊 Gráfico guardado: {filename}")
    plt.close()

def analisis_supervivencia(df):
    """Analiza la relación entre variables y la supervivencia (Solo Train)."""
    print("--- ANÁLISIS DE SUPERVIVENCIA (Solo Train) ---")
    
    # Correlación numérica
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Mapa de Calor de Correlaciones")
    plt.savefig("mapa_correlacion_train.png")
    print("📊 Gráfico guardado: mapa_correlacion_train.png")
    plt.close()
    
    # Supervivencia por Sexo
    print("\n>> Tasa de Supervivencia por Sexo:")
    print(df.groupby('Sex')['Survived'].mean())
    
    # Supervivencia por Clase
    print("\n>> Tasa de Supervivencia por Clase (Pclass):")
    print(df.groupby('Pclass')['Survived'].mean())
    
    # Gráfico de Supervivencia combinada
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)
    plt.title("Supervivencia por Clase y Sexo")
    plt.savefig("supervivencia_sexo_clase.png")
    print("📊 Gráfico guardado: supervivencia_sexo_clase.png")
    plt.close()

def main():
    train_df, test_df = cargar_datos()
    
    if train_df is not None:
        # 1. Análisis de Train
        analisis_estructura(train_df, "TRAIN")
        estadisticas_descriptivas(train_df, "TRAIN")
        plot_distribuciones_numericas(train_df, "train")
        plot_categoricas(train_df, "train")
        
        # 2. Análisis Específico de Supervivencia (Target)
        analisis_supervivencia(train_df)
        
        print("\n" + "="*50 + "\n")
        
        # 3. Análisis de Test (Breve revisión para consistencia)
        analisis_estructura(test_df, "TEST")
        estadisticas_descriptivas(test_df, "TEST")
        # Opcional: Gráficos para test también
        # plot_distribuciones_numericas(test_df, "test")

if __name__ == "__main__":
    main()