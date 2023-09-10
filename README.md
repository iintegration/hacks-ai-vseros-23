## Проект "Модель склонности клиента к приобретению машиноместа" от компании Самолёт
<!-- <a href="">Команда ИИнтеграция</a> -->

### Наша команда `ИИнтеграция`
<ul>
    <li><a href = "https://t.me/sherokiddo">Кирилл Брагин - Тимлид</a></li>
    <li><a href = "https://t.me/denisadminch">Денис Агапитов - Ml-engineer</a></li>
    <li><a href = "https://t.me/YarKo9_9">Ярослав Колташев - Ml-engineer</a></li>
    <li><a href = "https://t.me/p_petrovskiy_02">Павел Петровский - Ml-engineer</a></li>
    <li><a href = "https://t.me/gabbhack">Никита Габбасов - MlOpls</a></li>
    <li><a href = "https://t.me/kai_Kane">Кирилл Резников - Front <b>(Отпуск)</b> </a></li>
</ul>


## Описание кейса
В рамках всероссийского хакатона "Цифрового прорыва" 2023 в Нижнем Новгороде необходимо было решить кейс "Модель склонности клиента к приобретению машиноместа".

> Ключевую роль в продажах играет эффективная целевая рассылка. Рассылки позволяют оперативно информировать клиентов об актуальных предложениях и сервисах компании. Однако каждая рассылка сопряжена с различными издержками, что может негативно сказаться на лояльности клиентов в долгосрочной перспективе. На основе больших данных о предыдущем опыте взаимодействия с клиентами участникам хакатона предстоит разработать модель, позволяющую прогнозировать вероятность покупки клиентами дополнительных услуг, в частности, приобретения машиномест в паркинге. Разработанное решение позволит компании снизить затраты и улучшить лояльность клиентов за счет более персонифицированного подхода.

## Подготовка

Перед проведением хакатона, изучили проблематику кейса и сразу было понятно, что данная задача относятся к задаче таргетирования.

Для данной задачи

* подготовлен кастомыный алгоритм кросс-валидации `BlockedTimeSeriesSplit` для данных, которые предположительно предоставят кейсосодержатель

* реализован алгоритм поиска гиперпараметров на основе `Optuna` с применением `BlockedTimeSeriesSplit`

* реализован алгоритм факторного анализа данных на PySpark

## Хакатон

Во время хакатона компания Самолёт предоставила нам данные в формате `csv`, в которой имеются данные на первое число
каждого месяца за 1,5 года. 

> Входные данные представляют собой наборы
признаков по клиентам на первое число
каждого месяца за 1,5 года (sample данных).
Целевой признак равен 1, если в следующие 3
месяца клиент купит машиноместо. После
покупки машиноместа клиент исключается из
наборов данных.
 
В результатом модель должна выдавать `predict_proba` по каждому `client_id`.

> Результатом предсказания модели должен стать
скор (значение от 0 до 1) по каждому клиенту —
вероятность, что клиент купит машиноместо в
следующие 3 месяца.
Оцениваться результат будет метрикой ROCAUC.


Ключевую роль в продажах играет эффективная целевая рассылка. Рассылки
позволяют оперативно информировать клиентов об актуальных предложениях и
сервисах компании.
Однако каждая рассылка сопряжена с издержками:
* финансовые затраты - на подготовку и доставку сообщений (SMS, email и т.д.);
* временные затраты маркетологов и продавцов;
* риск раздражения получателей частыми сообщениями.
Это может негативно сказаться на лояльности клиентов в долгосрочной перспективе.
Поэтому компания хочет максимизировать отдачу от рассылок, ориентируясь только
на наиболее заинтересованных и потенциально ориентированных на покупку
клиентов.


## Решение

### Загрузка модель

Финальная модель находиться в папке `ML` под названием `model`

Чтобы загрузить модель необходимо сделать

```py
model = mlflow.sklearn.load_model('ML\\model')
```

Фичи которые используются для предсказания находятся в папке `ML` под названием `columns.json`

<hr>

### Таблица применения различного семплирования

`len_datas` - количество дат, которые присутствуют в тренировочной выборке

`samples` - отношение единиц к нулям в тренировочной выборке  `1:samples`

`pr_auc` - значение метрики Precision Recall AUC

`roc_auc` - значение метрики ROC AUC

|len_dates|samples|pr_auc|roc_auc|
| -------- | -------- | -------- | -------- |
|5 |6 | 0.221433 | 0.918495|
|5 | 8 | 0.258819 | 0.913699|
|2 | 2 | 0.229737 | 0.912241|
|2 | 10 | 0.272937 | 0.911979|
|6 | 3 | 0.248723 | 0.911783|
|2 | 1 | 0.221453 | 0.911698|
|5 | 9 | 0.238714 | 0.911469|
|5 | 10 | 0.184477 | 0.909631|


## Streamlit

### Установка зависимостей

```cmd
pip install -r ML/model/requirements.txt
pip install streamlit matplotlib streamlit-shap shap
```

### Запуск

```cmd
streamlit run streamlit/main.py
```
