#!/bin/bash

# Обновление списка пакетов
sudo apt-get update

export DEBIAN_FRONTEND=noninteractive

# Установка необходимых пакетов
sudo apt install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" nvidia-driver-550-server
sudo apt install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" nvidia-utils-550-server
sudo apt install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" python3-pip
sudo apt install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" python3-venv

# Создание директории для виртуальных окружений
mkdir -p environments

# Создание виртуального окружения
python3 -m venv environments/hack

# Активация виртуального окружения
source environments/hack/bin/activate

# Установка Jupyter и основных библиотек
sudo pip install jupyter
sudo pip install numpy pandas scikit-learn torch transformers
sudo pip install -r requirements.txt



# Библиотеки для ML и NLP

echo "Установка завершена."
