import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Config (más ligero para Codespaces)
IMG_SIZE = (160, 160)
BATCH_SIZE = 4
EPOCHS = 3


train_dir = "data/raw/train"
test_dir  = "data/raw/test"

# Generadores
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Clases:", train_data.class_indices)

# Modelo pequeño (sí corre en Codespaces)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Callbacks
os.makedirs("models", exist_ok=True)
ckpt = ModelCheckpoint("models/best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
early = EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True, verbose=1)

# Entrenamiento
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[ckpt, early]
)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[ckpt, early]
)


# Evaluación
test_loss, test_acc = model.evaluate(test_data, verbose=1)
print(f"✅ Test accuracy: {test_acc:.4f}")

# Guardar
model.save("models/final_model.keras")
print("✅ Guardado en models/best_model.keras y models/final_model.keras")

