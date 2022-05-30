# ML in prod: Homework 2

## How to start docker
Build docker image

```bash
sudo docker build -t potatochka/online_inference:v1 .
```

or pull this image from my repo

```bash
sudo docker pull potatochka/online_inference:v1
```

then run container
```bash
sudo docker run -p 8080:8080 potatochka/online_inference:v1
```

## Test the project

```bash
python3 -m pytest -v
```

## Create requests
```bash
python3 request.py 
```

## Optimization

Для уменьшения веса докер изображения:

1. взял в качетсве окружения`3.6-slim-stretch` (`96MB`) вроде один из самых легковесных питонов
2. Удалил все ненужные файлы для проекта, до этого пушил весь репозиторий
3. В requirments оставил только нужные пакеты

Вес изображения `559MB`.

## Самооценка

### Основная часть

1. Оберните inference вашей модели в rest сервис на FastAPI, должен быть endpoint /predict (3 балла) - сделано

2. Напишите endpoint /health (1 балл), должен возращать 200, если ваша модель готова к работе (такой чек особенно актуален если делаете доп задание про скачивание из хранилища) - сделано

3. Напишите unit тест для /predict (3 балла) (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/) - сделано

4. Напишите скрипт, который будет делать запросы к вашему сервису -- 2 балла - сделано

5. Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балл) - сделано

6. опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (+2 балла) - сделано

7. напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель (1 балл) Убедитесь, что вы можете протыкать его скриптом из пункта 3 - сделано

8. проведите самооценку(распишите в реквесте какие пункты выполнили и на сколько баллов, укажите сумму баллов) -- 1 балл - сделано

### Дополнительная часть:

1. Ваш сервис скачивает модель из S3 или любого другого хранилища при старте, путь для скачивания передается через переменные окружения (+2 доп балла) - не сделал
2. Оптимизируйте размер docker image (+2 доп балла) (опишите в readme.md что вы предприняли для сокращения размера и каких результатов удалось добиться -- должно получиться мини исследование, я сделал тото и получился такой-то результат) - оптимизировал как мог 
3. Сделайте валидацию входных данных (например, порядок колонок не совпадает с трейном, типы, допустимые максимальные и минимальные значения, проявите фантазию, это доп. баллы, проверка не должна быть тривиальной) (вы можете сохранить вместе с моделью доп информацию, о структуре входных данных, если это нужно) -- 2 доп балла https://fastapi.tiangolo.com/tutorial/handling-errors/ -- возращайте 400, в случае, если валидация не пройдена - сделано, но валидация довольно тривиальная, без особого воображения 

Сумма - 21 балл
