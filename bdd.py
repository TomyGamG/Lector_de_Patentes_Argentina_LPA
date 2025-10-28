import itertools
import csv
import gzip
from tqdm import tqdm  # librería para barra de progreso

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def mercosur_generator():
    """Genera patentes nuevas: AA000AA"""
    for a, b in itertools.product(LETTERS, repeat=2):
        for num in range(1000):
            middle = f"{num:03d}"
            for c, d in itertools.product(LETTERS, repeat=2):
                yield f"{a}{b}{middle}{c}{d}"

def vieja_generator():
    """Genera patentes viejas: AAA000"""
    for a, b, c in itertools.product(LETTERS, repeat=3):
        for num in range(1000):
            yield f"{a}{b}{c}{num:03d}"

def write_all_plates_to_csv_gzip(filename='patentes_argentinas.csv.gz', chunk_size=100000):
    """
    Escribe patentes nuevas y viejas en un CSV comprimido gzip.
    Muestra barra de progreso aproximada por tipo.
    """
    # conteos aproximados para porcentaje
    total_nueva = 26**4 * 1000  # 456,976,000
    total_vieja = 26**3 * 1000  # 17,576,000

    with gzip.open(filename, mode='wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patente', 'tipo'])
        # NUEVAS
        chunk = []
        print("Escribiendo patentes nuevas...")
        for i, plate in enumerate(tqdm(mercosur_generator(), total=total_nueva), start=1):
            chunk.append([plate, 'nueva'])
            if i % chunk_size == 0:
                writer.writerows(chunk)
                chunk = []
        if chunk:
            writer.writerows(chunk)

        # VIEJAS
        chunk = []
        print("Escribiendo patentes viejas...")
        for i, plate in enumerate(tqdm(vieja_generator(), total=total_vieja), start=1):
            chunk.append([plate, 'vieja'])
            if i % chunk_size == 0:
                writer.writerows(chunk)
                chunk = []
        if chunk:
            writer.writerows(chunk)

if __name__ == "__main__":
    write_all_plates_to_csv_gzip('patentes_argentinas.csv.gz', chunk_size=200000)
