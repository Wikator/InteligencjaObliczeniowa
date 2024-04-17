from keras.applications import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Załaduj pre-trained model Xception
base_model = Xception(weights='imagenet', include_top=False)

# Dodaj warstwy klasyfikatora na wierzch modelu
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Możesz dostosować liczbę neuronów w warstwie Dense
predictions = Dense(2, activation='softmax')(x)  # Warstwa wyjściowa dla dwóch klas: koty i psy

# Zestawienie modelu
model = Model(inputs=base_model.input, outputs=predictions)

# Zamroź wszystkie warstwy pre-trained modelu
for layer in base_model.layers:
    layer.trainable = False

# Kompilacja modelu
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3  # Ustawienie podziału na zbiory treningowy i walidacyjny
)

# Wczytaj obrazy z katalogu z podziałem na zbiory treningowy i walidacyjny
train_generator = train_datagen.flow_from_directory(
    'dataset/train',  # Ścieżka do katalogu z obrazami
    target_size=(299, 299),  # Rozmiar obrazów
    batch_size=32,
    subset='training'  # Określenie, że chcemy uzyskać generator danych dla zbioru treningowego
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/validation',  # Ścieżka do katalogu z obrazami
    target_size=(299, 299),  # Rozmiar obrazów
    batch_size=32,
    subset='validation'  # Określenie, że chcemy uzyskać generator danych dla zbioru walidacyjnego
)

# Sprawdź, czy dane zostały poprawnie wczytane
print(train_generator.classes)
print(validation_generator.classes)

# Trening modelu
model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=10,  # Możesz dostosować liczbę epok
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size)

# Po treningu możesz odblokować niektóre warstwy konwolucyjne i kontynuować trening
# np. odblokowanie ostatnich kilku warstw konwolucyjnych

for layer in model.layers[:100]:
    layer.trainable = False
for layer in model.layers[100:]:
    layer.trainable = True

# Kontynuacja treningu modelu
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size)
