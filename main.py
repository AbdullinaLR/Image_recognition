import os
from PIL import Image

# Путь к вашему каталогу с изображениями
image_dir = r"C:\Users\owlwi\OneDrive\Документы\сова\учеба\ии\medication"

# Желаемый размер изображения
desired_size = (64, 64)

for dirname in os.listdir(image_dir):
    dir_path = os.path.join(image_dir, dirname)
    for filename in os.listdir(dir_path):
        if filename.endswith(".png"):  # Проверка, что файл - изображение формата PNG
            image_path = os.path.join(dir_path, filename)
            image = Image.open(image_path)

            # Масштабирование изображения до желаемого размера
            image = image.resize(desired_size)

            # Сохранение измененного изображения обратно
            image.save(os.path.join(dir_path, filename))

print("Переформатирование завершено")
