import itertools
import csv
import gzip
from tqdm import tqdm

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Última patente conocida en 2025
ULTIMA_SERIE = ('A', 'H', 100, 'A', 'A')  # Representa AH100AA

def mercosur_generator_limit():
    """
    Genera patentes nuevas (AA000AA) hasta AH100AA inclusive.
    """
    for a, b in itertools.product(LETTERS, repeat=2):
        for num in range(1000):
            for c, d in itertools.product(LETTERS, repeat=2):
                yield f"{a}{b}{num:03d}{c}{d}"

                # Detener cuando lleguemos a la última patente real
                if (a, b, num, c, d) == ULTIMA_SERIE:
                    return

def vieja_generator():
    """
    Genera patentes viejas (AAA000).
    """
    for a, b, c in itertools.product(LETTERS, repeat=3):
        for num in range(1000):
            yield f"{a}{b}{c}{num:03d}"

def write_limited_plates_to_csv(filename='patentes_argentinas_limitadas.csv.gz', chunk_size=100000):
    """
    Escribe las patentes (viejas + nuevas hasta AH100AA) en un CSV comprimido gzip.
    Muestra barra de progreso.
    """
    print("Calculando cantidad de patentes nuevas hasta AH100AA...")

    # Calcular la cantidad de nuevas a generar hasta AH100AA
    pos_a = LETTERS.index('A')
    pos_b = LETTERS.index('H')
    total_nueva = (pos_a * 26 + pos_b) * 1000 * 26 * 26 + 100 * 26 * 26
    total_vieja = 26**3 * 1000
    total = total_nueva + total_vieja

    with gzip.open(filename, mode='wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patente', 'tipo'])

        print("Generando patentes nuevas (hasta AH100AA)...")
        chunk = []
        for plate in tqdm(mercosur_generator_limit(), total=total_nueva):
            chunk.append([plate, 'nueva'])
            if len(chunk) >= chunk_size:
                writer.writerows(chunk)
                chunk = []
        if chunk:
            writer.writerows(chunk)

        print("Generando patentes viejas...")
        chunk = []
        for plate in tqdm(vieja_generator(), total=total_vieja):
            chunk.append([plate, 'vieja'])
            if len(chunk) >= chunk_size:
                writer.writerows(chunk)
                chunk = []
        if chunk:
            writer.writerows(chunk)

    print(f"✅ Archivo generado: {filename}")

if __name__ == "__main__":
    write_limited_plates_to_csv('patentes_argentinas_limitadas.csv.gz', chunk_size=200000)
