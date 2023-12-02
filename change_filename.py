import os

# Путь к папке с изображениями
folder_path = 'data'

# Получаем список файлов в папке
files = os.listdir(folder_path)

# Счетчик для номерации файлов
counter = 1

# Проходим по всем файлам в папке
for file_name in files:
    # Проверяем, является ли файл изображением (можно добавить другие форматы изображений)
    if file_name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # Если файл .jpeg, то удаляем его
        if file_name.endswith('.jpeg'):
            # Формируем полный путь к файлу
            file_path = os.path.join(folder_path, file_name)

            # Удаляем файл
            os.remove(file_path)
        else:
            # Получаем расширение файла
            file_ext = os.path.splitext(file_name)[1]

            # Формируем новое имя файла
            new_name = f'data{counter}{file_ext}'

            # Полный путь к старому файлу
            old_path = os.path.join(folder_path, file_name)

            # Полный путь к новому файлу
            new_path = os.path.join(folder_path, new_name)

            # Переименовываем файл
            os.rename(old_path, new_path)

            # Увеличиваем счетчик
            counter += 1
