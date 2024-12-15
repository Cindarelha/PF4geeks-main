from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def depurar_ingredientes_eficiente(frecuencia_ingredientes, similitud_umbral=0.85, output_file='../data/raw/ingredientes_aptos_depurados.csv'):
    """
    Depura la lista de ingredientes aptos eliminando valores no deseados y agrupando ingredientes similares.
    Utiliza un enfoque basado en TF-IDF y similitud de coseno para agrupar ingredientes.

    Args:
        frecuencia_ingredientes (pd.DataFrame): DataFrame con ingredientes y sus frecuencias.
        similitud_umbral (float): Umbral de similitud (0-1) para considerar ingredientes como equivalentes.
        output_file (str): Nombre del archivo CSV de salida.

    Returns:
        None
    """
    # Filtrar valores no deseados (valores numéricos, nan, vacíos, etc.)
    frecuencia_ingredientes = frecuencia_ingredientes[
        frecuencia_ingredientes['Ingrediente'].apply(
            lambda x: isinstance(x, str) and not x.isdigit() and x.lower() != 'nan' and len(x.strip()) > 1
        )
    ]
    
    # Normalizar ingredientes (minúsculas y eliminar espacios innecesarios)
    frecuencia_ingredientes['Ingrediente_Normalizado'] = frecuencia_ingredientes['Ingrediente'].str.lower().str.strip()

    # Vectorizar los ingredientes usando TF-IDF
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(frecuencia_ingredientes['Ingrediente_Normalizado'])

    # Calcular similitud de coseno entre los ingredientes
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Crear agrupaciones basadas en similitud
    agrupaciones = {}
    usados = set()

    for idx, ingrediente in enumerate(frecuencia_ingredientes['Ingrediente_Normalizado']):
        if idx in usados:
            continue
        similares = np.where(cosine_sim[idx] >= similitud_umbral)[0]
        base = ingrediente
        agrupaciones[base] = [frecuencia_ingredientes['Ingrediente_Normalizado'].iloc[i] for i in similares]
        usados.update(similares)

    # Crear un mapeo de ingredientes a términos representativos
    mapeo = {sim: base for base, grupo in agrupaciones.items() for sim in grupo}

    # Reemplazar ingredientes en el DataFrame original
    frecuencia_ingredientes['Ingrediente_Depurado'] = frecuencia_ingredientes['Ingrediente_Normalizado'].map(mapeo)

    # Consolidar las frecuencias de los ingredientes depurados
    ingredientes_depurados = (
        frecuencia_ingredientes.groupby('Ingrediente_Depurado')['Frecuencia']
        .sum()
        .reset_index()
        .sort_values(by='Frecuencia', ascending=False)
    )

    # Guardar el resultado en un archivo CSV
    ingredientes_depurados.to_csv(output_file, index=False)
    print(f"Archivo CSV generado con los ingredientes depurados: {output_file}")

# Llamar a la función para depurar los ingredientes
depurar_ingredientes_eficiente(frecuencia_ingredientes, similitud_umbral=0.85, output_file='../data/raw/ingredientes_aptos_depurados.csv')
