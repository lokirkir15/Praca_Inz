import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
from datetime import datetime

DATASET_ROOT = "FER2013"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")
OUT_DIR   = "outputs_cnn"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42
EPOCHS = 40

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="training",
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Klasy:", class_names)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE

def configure(ds, training=False):
    if training:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
        ])
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_ds = configure(train_ds, training=True)
val_ds   = configure(val_ds, training=False)
test_ds  = configure(test_ds, training=False)

all_labels = []
for _, y in train_ds.unbatch():
    all_labels.append(np.argmax(y.numpy()))
all_labels = np.array(all_labels)

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=all_labels
)
class_weights = dict(enumerate(class_weights_arr))
print("Class weights:", class_weights)

def build_cnn(input_shape=(48,48,1), classes=7):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.SpatialDropout2D(0.20)(x)
        return x

    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)

    x = tf.keras.layers.Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn_fer2013_final")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model

model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), classes=num_classes)
model.summary()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ckpt_path = os.path.join(OUT_DIR, f"cnn_best_{timestamp}.keras")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoka")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("Epoka")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, f"history_{timestamp}.png")
plt.savefig(plot_path, dpi=140)
plt.close()
print(f"Zapisano wykres historii: {plot_path}")

test_images = []
test_true_idx = []

for batch_x, batch_y in test_ds:
    test_images.append(batch_x.numpy())
    test_true_idx.append(np.argmax(batch_y.numpy(), axis=1))

test_images = np.concatenate(test_images, axis=0)
test_true_idx = np.concatenate(test_true_idx, axis=0)

probs = model.predict(test_ds, verbose=1)
pred_idx = probs.argmax(axis=1)

acc = (pred_idx == test_true_idx).mean() * 100
print(f"\nAccuracy na teście: {acc:.2f}%")

cm = confusion_matrix(test_true_idx, pred_idx)
report = classification_report(
    test_true_idx, pred_idx, target_names=class_names, digits=4
)
print("\nClassification report:\n", report)

np.savetxt(os.path.join(OUT_DIR, f"confusion_matrix_{timestamp}.csv"),
           cm, fmt="%d", delimiter=",")
with open(os.path.join(OUT_DIR, f"classification_report_{timestamp}.txt"),
          "w", encoding="utf-8") as f:
    f.write(report)
    f.write(f"\nAccuracy: {acc:.2f}%\n")

test_filepaths = []
for root, _, files in os.walk(TEST_DIR):
    files = [f for f in files if f.lower().endswith((".jpg", ".png"))]
    files.sort()
    for f in files:
        test_filepaths.append(os.path.join(root, f))

idx2name = {i: name for i, name in enumerate(class_names)}

rows = []
for i, fp in enumerate(test_filepaths):
    true_label = idx2name[test_true_idx[i]]
    pred_label = idx2name[pred_idx[i]]
    true_prob = probs[i, test_true_idx[i]] * 100.0
    row = {
        "plik": fp,
        "emocja_prawdziwa": true_label,
        "emocja_predykcja": pred_label,
        "zgodność": "TAK" if true_label == pred_label else "NIE",
        "prawd_emocji_prawidłowej_%": round(float(true_prob), 2),
    }
    for j, name in enumerate(class_names):
        row[f"{name}_%"] = round(float(probs[i, j] * 100.0), 2)
    rows.append(row)

csv_path = os.path.join(OUT_DIR, f"cnn_results_{timestamp}.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Zapisano wyniki testu do CSV: {csv_path}")

fig = plt.figure(figsize=(14, 8))
shown = 0
picked_for = set()
for i, fp in enumerate(test_filepaths):
    true_label = idx2name[test_true_idx[i]]
    if true_label in picked_for:
        continue
    img = tf.keras.utils.load_img(fp, color_mode="grayscale", target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img).astype("uint8").squeeze(-1)
    shown += 1
    plt.subplot(2, 4, shown)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(
        f"Prawdziwa: {true_label}\n"
        f"Rozpoznana: {idx2name[pred_idx[i]]}\n"
        f"Prawd. prawidłowej: {float(probs[i, test_true_idx[i]]*100):.2f}%"
    )
    picked_for.add(true_label)
    if shown == min(7, num_classes):
        break
plt.tight_layout()
ex_path = os.path.join(OUT_DIR, f"examples_{timestamp}.png")
plt.savefig(ex_path, dpi=140)
plt.close()
print(f"Zapisano przykładowe obrazy: {ex_path}")

print("\nEtap 2 (CNN) zakończony.")
