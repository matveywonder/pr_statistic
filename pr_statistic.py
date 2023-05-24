#!/usr/bin/env python
# coding: utf-8

# ### Откройте файл с данными и изучите общую информацию

# In[1]:


import pandas as pd


# Открываем файл `/datasets/calls.csv`, сохраните датафрейм в переменную `calls`.

# In[2]:



calls = pd.read_csv('/datasets/calls.csv')


# Выводим первые 5 строк датафрейма `calls`.

# In[3]:


calls.head(5)


# Выводим основную информацию для датафрейма `calls` с помощью метода `info()`.

# In[4]:


calls.info()


# С помощью метода `hist()` выводим гистограмму для столбца с продолжительностью звонков. 

# In[5]:


calls['duration'].hist();


# Открываем файл `/datasets/internet.csv`, сохраните датафрейм в переменную `sessions`.

# In[6]:


sessions = pd.read_csv('/datasets/internet.csv')


# Выводим первые 5 строк датафрейма `sessions`.

# In[7]:


sessions.head()


# Выводим основную информацию для датафрейма `sessions` с помощью метода `info()`. 

# In[8]:


sessions.info()


# С помощью метода `hist()` выводим гистограмму для столбца с количеством потраченных мегабайт.

# In[9]:


sessions['mb_used'].hist()


# Открываем файл `/datasets/messages.csv`, сохраните датафрейм в переменную `messages`.

# In[10]:


messages = pd.read_csv('/datasets/messages.csv')


# Выводим первые 5 строк датафрейма `messages`.

# In[11]:


messages.head()


# Выводим основную информацию для датафрейма `messages` с помощью метода `info()`. 

# In[12]:


messages.info()


# Открываем файл `/datasets/tariffs.csv`, сохраните датафрейм в переменную `tariffs`.

# In[13]:


tariffs = pd.read_csv('/datasets/tariffs.csv')


# Выводим весь датафрейм `tariffs`.

# In[14]:


tariffs


# Выводим основную информацию для датафрейма `tariffs` с помощью метода `info()`.

# In[15]:


tariffs.info()


# Открываем файл `/datasets/users.csv`, сохраните датафрейм в переменную `users`.

# In[16]:


users = pd.read_csv('/datasets/users.csv')


# Выводим первые 5 строк датафрейма `users`.

# In[17]:


users.head()


# Выводим основную информацию для датафрейма `users` с помощью метода `info()`.

# In[18]:


users.info()


# ### Подготовка данных

# **Задание 18.**  Приводим столбцы
# 
# - `reg_date` из таблицы `users`
# - `churn_date` из таблицы `users`
# - `call_date` из таблицы `calls`
# - `message_date` из таблицы `messages`
# - `session_date` из таблицы `sessions`
# 
# к новому типу с помощью метода `to_datetime()`.

# In[19]:


users['reg_date'] = pd.to_datetime(users['reg_date'], format='%Y.%m.%d %H:%M:%S')
users['churn_date'] = pd.to_datetime(users['churn_date'], format='%Y.%m.%d %H:%M:%S')
calls['call_date'] = pd.to_datetime(calls['call_date'], format='%Y.%m.%d %H:%M:%S')
messages['message_date'] = pd.to_datetime(messages['message_date'], format='%Y.%m.%d %H:%M:%S')
sessions['session_date'] = pd.to_datetime(sessions['session_date'], format='%Y.%m.%d %H:%M:%S')


# В данных есть звонки с нулевой продолжительностью. Это не ошибка: нулями обозначены пропущенные звонки, поэтому их не нужно удалять.
# 
# Однако в столбце `duration` датафрейма `calls` значения дробные. Округляем значения столбца `duration` вверх с помощью метода `numpy.ceil()` и приведите столбец `duration` к типу `int`.

# In[20]:


import numpy as np

calls['duration'] = np.ceil(calls['duration']).astype(int)


# Удаляем столбец `Unnamed: 0` из датафрейма `sessions`. 

# In[21]:


sessions = sessions.drop(columns='Unnamed: 0')


# Создаем столбец `month` в датафрейме `calls` с номером месяца из столбца `call_date`.

# In[22]:


calls['month'] = pd.DatetimeIndex(calls['call_date']).month


# Создаем столбец `month` в датафрейме `messages` с номером месяца из столбца `message_date`.

# In[23]:


messages['month'] = pd.DatetimeIndex(messages['message_date']).month


# Создаем столбец `month` в датафрейме `sessions` с номером месяца из столбца `session_date`.

# In[24]:


sessions['month'] = pd.DatetimeIndex(sessions['session_date']).month


# Считаем количество сделанных звонков разговора для каждого пользователя по месяцам.

# In[25]:


calls_per_month = calls.groupby(['user_id', 'month']).agg(calls=('duration', 'count'))


# In[26]:


calls_per_month.head(30)


# Считаем количество израсходованных минут разговора для каждого пользователя по месяцам и сохраняем в переменную `minutes_per_month`. Вам понадобится
# 
# Выводим первые 30 строчек `minutes_per_month`.

# In[27]:


minutes_per_month = calls.groupby(['user_id', 'month']).agg(minutes=('duration', 'sum'))


# In[28]:


minutes_per_month.head(30)


# Считаем количество отправленных сообщений по месяцам для каждого пользователя и сохраняем в переменную `messages_per_month`. Вам понадобится
# 
# Выводим первые 30 строчек `messages_per_month`.

# In[29]:


messages_per_month = messages.groupby(['user_id', 'month']).agg(messages=('message_date', 'count'))


# In[30]:


messages_per_month.head(30)


# Считаем количество потраченных мегабайт по месяцам для каждого пользователя и сохраняем в переменную `sessions_per_month`. 

# In[31]:


sessions_per_month = sessions.groupby(['user_id', 'month']).agg({'mb_used': 'sum'})


# In[32]:


sessions_per_month.head(30)


# ### Анализ данных и подсчёт выручки

# Объединяем все посчитанные выше значения в один датафрейм `user_behavior`.

# In[33]:


users['churn_date'].count() / users['churn_date'].shape[0] * 100


# Расторгли договор 7.6% клиентов из датасета

# In[34]:


user_behavior = calls_per_month    .merge(messages_per_month, left_index=True, right_index=True, how='outer')    .merge(sessions_per_month, left_index=True, right_index=True, how='outer')    .merge(minutes_per_month, left_index=True, right_index=True, how='outer')    .reset_index()    .merge(users, how='left', left_on='user_id', right_on='user_id')
user_behavior.head()


# Проверим пропуски в таблице `user_behavior` после объединения:

# In[35]:


user_behavior.isna().sum()


# Заполним образовавшиеся пропуски в данных:

# In[36]:


user_behavior['calls'] = user_behavior['calls'].fillna(0)
user_behavior['minutes'] = user_behavior['minutes'].fillna(0)
user_behavior['messages'] = user_behavior['messages'].fillna(0)
user_behavior['mb_used'] = user_behavior['mb_used'].fillna(0)


# Присоединяем информацию о тарифах

# In[37]:


# переименование столбца tariff_name на более простое tariff

tariffs = tariffs.rename(
    columns={
        'tariff_name': 'tariff'
    }
)


# In[38]:


user_behavior = user_behavior.merge(tariffs, on='tariff')


# Считаем количество минут разговора, сообщений и мегабайт, превышающих включённые в тариф
# 

# In[39]:


user_behavior['paid_minutes'] = user_behavior['minutes'] - user_behavior['minutes_included']
user_behavior['paid_messages'] = user_behavior['messages'] - user_behavior['messages_included']
user_behavior['paid_mb'] = user_behavior['mb_used'] - user_behavior['mb_per_month_included']

for col in ['paid_messages', 'paid_minutes', 'paid_mb']:
    user_behavior.loc[user_behavior[col] < 0, col] = 0


# Переводим превышающие тариф мегабайты в гигабайты и сохраняем в столбец `paid_gb`

# In[40]:


user_behavior['paid_gb'] = np.ceil(user_behavior['paid_mb'] / 1024).astype(int)


# Считаем выручку за минуты разговора, сообщения и интернет

# In[41]:


user_behavior['cost_minutes'] = user_behavior['paid_minutes'] * user_behavior['rub_per_minute']
user_behavior['cost_messages'] = user_behavior['paid_messages'] * user_behavior['rub_per_message']
user_behavior['cost_gb'] = user_behavior['paid_gb'] * user_behavior['rub_per_gb']


# Считаем помесячную выручку с каждого пользователя, она будет храниться в столбце `total_cost`

# In[42]:


user_behavior['total_cost'] =       user_behavior['rub_monthly_fee']    + user_behavior['cost_minutes']    + user_behavior['cost_messages']    + user_behavior['cost_gb']


# Датафрейм `stats_df` для каждой пары «месяц — тариф» будет хранить основные характеристики

# In[43]:


# сохранение статистических метрик для каждой пары месяц-тариф
# в одной таблице stats_df (среднее значение, стандартное отклонение, медиана)

stats_df = user_behavior.pivot_table(
            index=['month', 'tariff'],\
            values=['calls', 'minutes', 'messages', 'mb_used'],\
            aggfunc=['mean', 'std', 'median']\
).round(2).reset_index()

stats_df.columns=['month', 'tariff', 'calls_mean', 'sessions_mean', 'messages_mean', 'minutes_mean',
                                     'calls_std',  'sessions_std', 'messages_std', 'minutes_std', 
                                     'calls_median', 'sessions_median', 'messages_median',  'minutes_median']

stats_df.head(10)


# Распределение среднего количества звонков по видам тарифов и месяцам

# In[44]:


import seaborn as sns

ax = sns.barplot(x='month',
            y='calls_mean',
            hue="tariff",
            data=stats_df,
            palette=['lightblue', 'blue'])

ax.set_title('Распределение количества звонков по видам тарифов и месяцам')
ax.set(xlabel='Номер месяца', ylabel='Среднее количество звонков');


# In[45]:


import matplotlib.pyplot as plt

user_behavior.groupby('tariff')['calls'].plot(kind='hist', bins=35, alpha=0.5)
plt.legend(['Smart', 'Ultra'])
plt.xlabel('Количество звонков')
plt.ylabel('Количество клиентов')
plt.show()


# Распределение средней продолжительности звонков по видам тарифов и месяцам

# In[46]:


ax = sns.barplot(x='month',
            y='minutes_mean',
            hue="tariff",
            data=stats_df,
            palette=['lightblue', 'blue'])

ax.set_title('Распределение продолжительности звонков по видам тарифов и месяцам')
ax.set(xlabel='Номер месяца', ylabel='Средняя продолжительность звонков');


# In[47]:


user_behavior[user_behavior['tariff'] =='smart']['minutes'].hist(bins=35, alpha=0.5, color='green')
user_behavior[user_behavior['tariff'] =='ultra']['minutes'].hist(bins=35, alpha=0.5, color='blue');


# Средняя длительность разговоров у абонентов тарифа Ultra больше, чем у абонентов тарифа Smart. В течение года пользователи обоих тарифов увеличивают среднюю продолжительность своих разговоров. Рост средней длительности разговоров у абонентов тарифа Smart равномерный в течение года. Пользователи тарифа Ultra не проявляют подобной линейной стабильности. Стоит отметить, что феврале у абонентов обоих тарифных планов наблюдались самые низкие показатели.

# Распределение среднего количества сообщений по видам тарифов и месяцам

# In[48]:


ax = sns.barplot(x='month',
            y='messages_mean',
            hue="tariff",
            data=stats_df,
            palette=['lightblue', 'blue']
)

ax.set_title('Распределение количества сообщений по видам тарифов и месяцам')
ax.set(xlabel='Номер месяца', ylabel='Среднее количество сообщений');


# In[49]:


user_behavior[user_behavior['tariff'] =='smart']['messages'].hist(bins=35, alpha=0.5, color='green')
user_behavior[user_behavior['tariff'] =='ultra']['messages'].hist(bins=35, alpha=0.5, color='blue');


# В среднем пользователи тарифа Ultra отправляют больше сообщений — почти на 20 сообщений больше, чем пользователи тарифа Smart. Количество сообщений в течение года на обоих тарифах растёт. Динамика по отправке сообщений схожа с тенденциями по длительности разговоров: в феврале отмечено наименьшее количество сообщений за год и пользователи тарифа Ultra также проявляют нелинейную положительную динамику.

# In[50]:


ax = sns.barplot(x='month',
            y='sessions_mean',
            hue='tariff',
            data=stats_df,
            palette=['lightblue', 'blue']
)

ax.set_title('Распределение количества потраченного трафика (Мб) по видам тарифов и месяцам')
ax.set(xlabel='Номер месяца', ylabel='Среднее количество мегабайт');


# Сравнение потраченных мегабайт среди пользователей тарифов Smart и Ultra

# In[51]:


user_behavior[user_behavior['tariff'] =='smart']['mb_used'].hist(bins=35, alpha=0.5, color='green')
user_behavior[user_behavior['tariff'] =='ultra']['mb_used'].hist(bins=35, alpha=0.5, color='blue');


# Меньше всего пользователи использовали интернет в январе, феврале и апреле. Чаще всего абоненты тарифа Smart тратят 15–17 Гб, а абоненты тарифного плана Ultra — 19–21 ГБ.

# ### Проверка гипотез

# Проверка гипотезы: средняя выручка пользователей тарифов «Ультра» и «Смарт» различается;
# 
# ```
# H_0: Выручка (total_cost) пользователей "Ультра" = выручка (total_cost) пользователей "Смарт"`
# H_a: Выручка (total_cost) пользователей "Ультра" ≠ выручка (total_cost) пользователей "Смарт"`
# alpha = 0.05
# ```

# In[52]:


from scipy import stats as st


# In[55]:


results = st.ttest_ind(
    user_behavior.loc[user_behavior.tariff == 'ultra', 'total_cost'],
    user_behavior.loc[user_behavior.tariff == 'smart', 'total_cost'], 
    equal_var=False)


alpha = 0.05

print('p-значение:', results.pvalue)
if results.pvalue < alpha:
    print('Отвергаем нулевую гипотезу')
else:
    print('Не получилось отвергнуть нулевую гипотезу') 


# Проверка гипотезы: средняя выручка с пользователей из Москвы отличается от выручки c пользователей других регионов; 
# 
# ```
# H_0: Выручка (total_cost) пользователей из Москвы = выручка (total_cost) пользователей не из Москвы`
# H_1: Выручка (total_cost) пользователей из Москвы ≠ выручка (total_cost) пользователей не из Москвы`
# alpha = 0.05
# ```

# In[56]:


results = st.ttest_ind(
    user_behavior.loc[user_behavior.city == 'Москва', 'total_cost'],
    user_behavior.loc[user_behavior.city != 'Москва', 'total_cost'], 
    equal_var=False)


alpha = 0.05

print('p-значение:', results.pvalue)
if results.pvalue < alpha:
    print('Отвергаем нулевую гипотезу')
else:
    print('Не получилось отвергнуть нулевую гипотезу') 


# In[ ]:




