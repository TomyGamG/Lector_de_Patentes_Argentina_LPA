import gzip
import csv

with gzip.open('patentes_argentinas_limitadas.csv.gz', mode='rt', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        print(row)
