from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import os
from PIL import Image
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Создание модели
model = Sequential([
    Conv2D(32, kernel_size=(5, 5), input_shape=(64, 64, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 классов для лекарств
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Папка с данными
data_dir = r'C:\Users\owlwi\OneDrive\Документы\сова\учеба\ии\medication'

# Размер изображений
image_size = (64, 64)

# Списки для данных и меток
data = []
labels = []

# Перебор всех папок (классов)
for class_folder in os.listdir(data_dir):
    class_folder_path = os.path.join(data_dir, class_folder)

    # Перебор всех изображений в папке
    for image_filename in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, image_filename)

        # Загрузка изображения и изменение его размера
        image = Image.open(image_path)
        image = image.resize(image_size)

        # Преобразование изображения в массив NumPy и нормализация пикселей
        image = image.convert('RGB')
        image = np.array(image, dtype=np.float32) / 255.0

        # Добавление изображения и метки в списки
        data.append(image)
        labels.append(class_folder)

# Преобразование меток в формат one-hot
label_map = {label: index for index, label in enumerate(set(labels))}


labels = [label_map[label] for label in labels]
labels = to_categorical(labels, num_classes=len(label_map))

# Преобразование списков в массивы NumPy
data = np.array(data)
labels = np.array(labels)

# Разделение данных на обучающий, валидационный и тестовый наборы

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.15, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.15,
                                                                  random_state=42)

# Обучение модели
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=90, batch_size=32)

# Оценка и тестирование модели
loss, accuracy = model.evaluate(test_data, test_labels)
model.save('image_recognition.h5')
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
