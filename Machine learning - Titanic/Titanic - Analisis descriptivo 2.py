import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estilo
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def cargar_datos():
    if not os.path.exists('train.csv'):
        print("❌ Error: Falta el archivo 'train.csv'.")
        return None
    return pd.read_csv('train.csv')

def clasificar_edad(edad):
    """Función auxiliar para crear los rangos de edad solicitados"""
    if pd.isna(edad):
        return "Desconocido"
    elif edad <= 14:
        return "Niño (0-14)"
    elif 15 <= edad <= 25:
        return "Adulto Joven (15-25)"
    elif 26 <= edad <= 60:
        return "Adulto (26-60)"
    else: # 61 o más
        return "Adulto Mayor (61+)"

def analisis_edad_detalle(df):
    """
    Analiza la supervivencia basada en rangos de edad personalizados:
    - 0-14: Niño
    - 18-25: Adulto joven
    - 26-60: Adulto
    - 61+: Adulto mayor
    """
    print("\n--- ANÁLISIS DETALLADO POR RANGO DE EDAD ---")
    
    # Creamos una copia para no afectar el dataframe original
    df_analisis = df.copy()
    
    # Aplicamos la clasificación
    df_analisis['Rango_Edad'] = df_analisis['Age'].apply(clasificar_edad)
    
    # Filtramos los 'Desconocido' para el gráfico (opcional, pero recomendado para limpieza)
    df_clean = df_analisis[df_analisis['Rango_Edad'] != 'Desconocido']
    
    # 1. Tabla de Estadísticas
    grupo = df_clean.groupby('Rango_Edad')['Survived'].agg(['count', 'mean'])
    grupo.columns = ['Total Pasajeros', 'Tasa Supervivencia']
    grupo['Tasa Supervivencia %'] = (grupo['Tasa Supervivencia'] * 100).round(2)
    
    # Ordenar lógicamente los índices
    orden_indices = ["Niño (0-14)", "Adulto Joven (15-25)", "Adulto (26-60)", "Adulto Mayor (61+)"]
    # Reindexar solo con los que existan en los datos
    indices_existentes = [i for i in orden_indices if i in grupo.index]
    grupo = grupo.loc[indices_existentes]
    
    print(grupo)
    
    # 2. Generar Gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Rango_Edad', y='Survived', data=df_clean, order=indices_existentes, palette='magma', errorbar=None)
    
    plt.title('Probabilidad de Supervivencia por Rango de Edad', fontsize=14)
    plt.xlabel('Grupo de Edad')
    plt.ylabel('Tasa de Supervivencia (0 a 1)')
    plt.ylim(0, 1) # Eje Y de 0% a 100%
    
    # Añadir las etiquetas de porcentaje encima de las barras
    for i, rango in enumerate(indices_existentes):
        valor = grupo.loc[rango, 'Tasa Supervivencia']
        plt.text(i, valor + 0.02, f"{valor*100:.1f}%", ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    filename = "supervivencia_por_rango_edad.png"
    plt.savefig(filename)
    print(f"\n📊 Gráfico guardado: {filename}")
    plt.close()

def main():
    train_df = cargar_datos()
    
    if train_df is not None:
        # Ejecutamos el análisis específico que pediste
        analisis_edad_detalle(train_df)
        
        # Puedes agregar aquí las otras funciones del código anterior si las necesitas
        # por ejemplo: analisis_estructura(train_df, "TRAIN")

if __name__ == "__main__":
    main()