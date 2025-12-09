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
OUT_DIR   = "outputs_resnet50"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE_SRC = (48, 48)
IMG_SIZE_NET = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS_FROZEN = 8
EPOCHS_FINE   = 15

LR_FROZEN = 1e-3
LR_FINE   = 1e-5

train_raw = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE_SRC,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="training",
)

val_raw = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE_SRC,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation",
)

class_names = train_raw.class_names
num_classes = len(class_names)
print("Klasy:", class_names)

test_raw = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE_SRC,
    shuffle=False,
)

all_labels = []
for _, y in train_raw.unbatch():
    all_labels.append(np.argmax(y.numpy()))
all_labels = np.array(all_labels)

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=all_labels
)
class_weights = dict(enumerate(class_weights_arr))
print("Class weights:", class_weights)

AUTOTUNE = tf.data.AUTOTUNE

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
    tf.keras.layers.RandomContrast(0.10),
], name="augmentation")

def to_resnet(x, y, training=False):
    x = tf.image.grayscale_to_rgb(x)
    x = tf.image.resize(x, IMG_SIZE_NET, method="bilinear")
    if training:
        x = augment(x, training=True)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x, y

train_ds = train_raw.map(lambda x, y: to_resnet(x, y, training=True),
                         num_parallel_calls=AUTOTUNE)
val_ds   = val_raw.map(lambda x, y: to_resnet(x, y, training=False),
                       num_parallel_calls=AUTOTUNE)
test_ds  = test_raw.map(lambda x, y: to_resnet(x, y, training=False),
                        num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE_NET[0], IMG_SIZE_NET[1], 3),
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE_NET[0], IMG_SIZE_NET[1], 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
)(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name="resnet50_fer2013")

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
    loss=loss_fn,
    metrics=["accuracy"],
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ckpt_path = os.path.join(OUT_DIR, f"resnet50_best_{timestamp}.keras")

callbacks_frozen = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6,
        restore_best_weights=True, verbose=1
    ),
]

hist1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=callbacks_frozen,
    class_weight=class_weights,
)

base.trainable = True
set_trainable = False
for layer in base.layers:
    if layer.name == "conv5_block1_out":
        set_trainable = True
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = set_trainable

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINE),
    loss=loss_fn,
    metrics=["accuracy"],
)

callbacks_fine = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6,
        restore_best_weights=True, verbose=1
    ),
]

hist2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_fine,
    class_weight=class_weights,
)

def plot_and_save(histories, out_png):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(acc, label="train acc")
    plt.plot(val_acc, label="val acc")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

plot_path = os.path.join(OUT_DIR, f"history_{timestamp}.png")
plot_and_save([hist1, hist2], plot_path)
print(f"Zapisano wykres historii: {plot_path}")

true_idx_list = []
for _, yb in test_raw:
    true_idx_list.append(np.argmax(yb.numpy(), axis=1))
true_idx = np.concatenate(true_idx_list, axis=0)

probs = model.predict(test_ds, verbose=1)
pred_idx = probs.argmax(axis=1)

acc = (pred_idx == true_idx).mean() * 100
print(f"\nAccuracy na teście: {acc:.2f}%")

cm = confusion_matrix(true_idx, pred_idx)
report = classification_report(true_idx, pred_idx,
                               target_names=class_names, digits=4)
print("\nClassification report:\n", report)

np.savetxt(
    os.path.join(OUT_DIR, f"confusion_matrix_{timestamp}.csv"),
    cm, fmt="%d", delimiter=","
)
with open(os.path.join(OUT_DIR, f"classification_report_{timestamp}.txt"),
          "w", encoding="utf-8") as f:
    f.write(report)
    f.write(f"\nAccuracy: {acc:.2f}%\n")

test_filepaths = []
for cls in test_raw.class_names:
    folder = os.path.join(TEST_DIR, cls)
    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".png"))]
    files.sort()
    for f in files:
        test_filepaths.append(os.path.join(folder, f))

idx2name = {i: n for i, n in enumerate(class_names)}
rows = []
for i, fp in enumerate(test_filepaths):
    true_label = idx2name[true_idx[i]]
    pred_label = idx2name[pred_idx[i]]
    true_prob = float(probs[i, true_idx[i]] * 100.0)
    row = {
        "plik": fp,
        "emocja_prawdziwa": true_label,
        "emocja_predykcja": pred_label,
        "zgodność": "TAK" if true_label == pred_label else "NIE",
        "prawd_emocji_prawidłowej_%": round(true_prob, 2),
    }
    for j, name in enumerate(class_names):
        row[f"{name}_%"] = round(float(probs[i, j] * 100.0), 2)
    rows.append(row)

csv_path = os.path.join(OUT_DIR, f"resnet50_results_{timestamp}.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Zapisano wyniki testu do CSV: {csv_path}")

fig = plt.figure(figsize=(14, 8))
shown, picked = 0, set()
for i, fp in enumerate(test_filepaths):
    true_label = idx2name[true_idx[i]]
    if true_label in picked:
        continue
    img = tf.keras.utils.load_img(
        fp, color_mode="grayscale", target_size=IMG_SIZE_SRC
    )
    img = tf.keras.utils.img_to_array(img).astype("uint8").squeeze(-1)
    shown += 1
    plt.subplot(2, 4, shown)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(
        f"Prawdziwa: {true_label}\n"
        f"Rozpoznana: {idx2name[pred_idx[i]]}\n"
        f"Prawd. prawidłowej: {float(probs[i, true_idx[i]]*100):.2f}%"
    )
    picked.add(true_label)
    if shown == min(7, num_classes):
        break

plt.tight_layout()
ex_path = os.path.join(OUT_DIR, f"examples_{timestamp}.png")
plt.savefig(ex_path, dpi=140)
plt.close()
print(f"Zapisano przykładowe obrazy: {ex_path}")

print("\nEtap 3 (ResNet50) zakończony.")
