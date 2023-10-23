from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tensorflow import keras
from PIL import Image
import numpy as np


def clicked():
    model = keras.models.load_model(r'C:\Users\owlwi\PycharmProjects\image_recognition\image_recognition.h5')
    image_path = filedialog.askopenfilename()
    image = Image.open(image_path)
    image = image.resize((64, 64))  # Убедитесь, что размер соответствует ожидаемому размеру модели
    image = image.convert('RGB')
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Добавьте размерность для батча

    # Классификация изображения
    predictions = model.predict(image)

    # Интерпретация результата
    class_index = np.argmax(predictions)  # Индекс класса с наивысшей вероятностью
    label_map = {
        9: 'диклофенак',
        8: 'ибупрофен',
        7: 'кагоцел',
        6: 'корвалол',
        5: 'нурофен',
        4: 'парацетамол',
        3: 'панкреатин',
        2: 'омепразол',
        1: 'хлоргексидин',
        0: 'цитрамон'

    }

    for label in label_map:
        if label == class_index:
            class_label = label_map[class_index]
    messagebox.showinfo('Результат анализа', f'Модель предсказывает класс: {class_label}')


window = Tk()
window.title("Расшифруем почерк")

window.geometry('400x300')
frame = Frame(window, padx=10, pady=10)
frame.pack(expand=True)

cal_btn = Button(frame, text='Выбрать файл для анализа', command=clicked)
cal_btn.grid(row=5, column=2)

window.mainloop()
