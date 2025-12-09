import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from deepface import DeepFace
from tqdm import tqdm

# Parametry
DATASET_PATH = "FER2013/test"
OUTPUT_CSV = "deepface_results.csv"
SAMPLES_PER_EMOTION = 2000

emotions = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
all_samples = []

for emotion in emotions:
    folder = os.path.join(DATASET_PATH, emotion)
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
    random.shuffle(images)
    selected = images[:SAMPLES_PER_EMOTION]
    for img_path in selected:
        all_samples.append((img_path, emotion))

print(f"Znaleziono {len(all_samples)} obrazów do analizy ({len(emotions)} emocji po maks. {SAMPLES_PER_EMOTION} zdjęć).")

results = []

for img_path, true_emotion in tqdm(all_samples, desc="Analiza emocji"):
    try:
        analysis = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
        data = analysis[0] if isinstance(analysis, list) else analysis
        dominant = data['dominant_emotion']
        probs = data['emotion']

        probs_percent = {k: round(v, 2) for k, v in probs.items()}

        correct = "TAK" if dominant == true_emotion else "NIE"
        correct_emotion_percent = probs_percent.get(true_emotion, 0.0)

        result = {
            "plik": img_path,
            "emocja_prawdziwa": true_emotion,
            "emocja_predykcja": dominant,
            "zgodność": correct,
            "prawd_emocji_prawidłowej_%": correct_emotion_percent
        }
        for emo, val in probs_percent.items():
            result[f"{emo}_%"] = val

        results.append(result)

    except Exception as e:
        print(f"Błąd dla {img_path}: {e}")

df = pd.DataFrame(results)
accuracy = round((df["zgodność"] == "TAK").mean() * 100, 2)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

with open(OUTPUT_CSV, "a", encoding="utf-8-sig") as f:
    f.write(f"\nŚrednia zgodność: {accuracy}%\n")

print(f"\nAnaliza zakończona. Wyniki zapisano w: {OUTPUT_CSV}")
print(f"Średnia zgodność (accuracy): {accuracy}%")

print("\nPrzykładowe wyniki:")

examples = []
for emotion in emotions:
    sample = df[df["emocja_prawdziwa"] == emotion].head(1)
    if not sample.empty:
        examples.append(sample.iloc[0])

plt.figure(figsize=(14, 8))
for i, ex in enumerate(examples):
    img = cv2.imread(ex["plik"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, i + 1)
    plt.imshow(img)
    plt.axis("off")
    percent = float(ex["prawd_emocji_prawidłowej_%"])
    plt.title(
        f"Prawdziwa: {ex['emocja_prawdziwa']}\n"
        f"Rozpoznana: {ex['emocja_predykcja']}\n"
        f"Prawd. prawidłowej: {percent:.2f}%"
    )
plt.tight_layout()
plt.show()
