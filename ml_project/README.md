# POTATOCHKA
Дз1 по курсу "Машинное обучение в продакшене" 
### Data
Already exist in folder data/raw because of small size
### Preresquistes

* [`Python 3`](https://www.python.org/)
* `virtualenv` (`pip install virtualenv`)

Create virtual envirenment and activate it
```bash
virtualenv venv
. venv/bin/activate
```
Install the modules
```bash
pip install -r requirements.txt
```
### EDA
Run
```bash
python3 data_explore/eda.py 
```
Script creates in folder ./data_explore markdown file with info about data
### Training

```bash
python3 build_model/train.py --config-name logreg_learn.yaml
python3 build_model/train.py --config-name boosting_learn.yaml
```

### Predict

```bash
python3 build_model/predict.py --config-name logreg_learn.yaml
python3 build_model/predict.py --config-name boosting_learn.yaml
```

### test transformer

```bash
 python3 -m pytest -v
```
### Структура проекта
```
└─ ml_project
   ├─ README.md
   ├─ configs
   │  ├─ boosting_learn.yaml
   │  └─ logreg_learn.yaml
   ├─ data
   │  └─ raw
   │     └─ heart_cleveland_upload.csv
   ├─ data_explore
   │  ├─eda.py
   │  ├─data_info.md
   │  └─hist.png
   ├─ models
   │  ├─ boosting
   │  │  ├─ metrics.json
   │  │  ├─ model.pkl
   │  │  ├─ predicts.csv
   │  │  └─ transform.pkl
   │  └─ logreg
   │     ├─ metrics.json
   │     ├─ model.pkl
   │     ├─ predicts.csv
   │     └─ transform.pkl
   ├─ notebooks
   │  ├─ eda.ipynb
   │  └─ train_model.ipynb
   ├─ utils
   │  ├─ read_config.py
   │  ├─ split.py
   │  ├─ train_model.py
   │  └─ transform_class.py
   └─ requirements.txt
```

### Архитектурные решения:

- В папке ./build_model лежат файлы с основными пайплайнами по обучению и предсказанию
- В ./configs лежат два примера конфигов, в которых определены основные переменные для запуска скриптов обучения
- В ./data/raw находится данные, которые использовались для обучения и предсказания
- В ./data_explore лежит eda данного проекта
- В ./models/{model_name} лежат метрики обученных можелей, а такж сохраненные веса
- Для конфигурирования использовалась `hydra`
- При запуске скриптов обучения создается папка outputs где сохраняются логи запусков
- Для сущностей из конфигов были написаны датаклассы содержатся в файле `./utils/read_config.py`
- Кастомный трансформер данных содержится в `./utils/transform_class.py`

### Оценка работы
0. В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. (1/1)
1. В пулл-реквесте проведена самооценка (1/1)
2. Выполнено EDA с использованием скрипта. (2/2)
3. Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкция по запуску (3/3)
4. Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3/3)
5. Проект имеет модульную структуру. (2/2)
6. Использованы логгеры (2/2)
7. Написаны тесты на отдельные модули и на прогон обучения и predict. (0/3)
8. Для тестов генерируются синтетические данные, приближенные к реальным. (0/2)
9. Обучение модели конфигурируется с помощью конфигов yaml. (3/3) 
10. Используются датаклассы для сущностей из конфига, а не голые dict (2/2)
11. Напишите кастомный трансформер и протестируйте его (3/3)
12. В проекте зафиксированы все зависимости (1/1)
13. Настроен CI для прогона тестов, линтера на основе github actions (0/3).

Дополнительные баллы:
- Используйте hydra для конфигурирования (3/3)
итого 26