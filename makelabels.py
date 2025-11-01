import os
import pandas as pd

folder = "dataset"
files = [f for f in os.listdir(folder) if f.endswith(".png")]

data = []
for f in files:
    plate = f.split("_")[0]  # etiqueta antes del primer '_'
    data.append([f, plate])

df = pd.DataFrame(data, columns=["filename", "label"])
df.to_csv("plates_labels.csv", index=False)

print("✅ CSV generado: plates_labels.csv")
