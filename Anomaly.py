import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
from sklearn.svm import OneClassSVM

df = pd.read_csv('ssh_anomaly_dataset.csv', encoding='utf-8')
df.drop(columns=['detail'], inplace=True)

# 1. считаем частоту выскакиваний IP
ip_counts = df['source_ip'].value_counts(normalize=True).to_dict()
df["ip_frequency"] = df['source_ip'].map(ip_counts)

# 1.1 переводим время в час суток и сортируем
df['timestamp'] = pd.to_datetime(df["timestamp"])
df['hour'] = df['timestamp'].dt.hour
df = df.sort_values('timestamp')

# 1.2 частота username
user_counts = df['username'].value_counts(normalize=True).to_dict()
df['username_frequency'] = df['username'].map(user_counts)

# 1.3 частота статуса
df['status_frequency'] = df['status'].map(df['status'].value_counts(normalize=True))

# 1.4 выделяем bash-команды 0 или 1
suspicious_commands = ['ls -la', 'whoami', 'free -m', 'pwd', 'df -h', 'uptime']
df['is_bash_command'] = df['status'].isin(suspicious_commands).astype(int)

# 1.5 еще один признак: сколько всего событий произошло с IP
df['ip_cumulative_events'] = df.groupby('source_ip').cumcount()

# 1.6 переводим строчки в 0 или 1
df['status_code'] = df['status'].apply(lambda x: 0 if x in ['success', 'normal_logout'] else 1)

# 1.7 еще: сколько раз этот IP ошибался
df['ip_cumulative_failures'] = df.groupby('source_ip')['status_code'].cumsum()

# 1.8 еще: процент ошибок от общего числа действий IP
# смысл: скорость/интенсивность ошибки
df['ip_failure_rate'] = df['ip_cumulative_failures'] / (df['ip_cumulative_events'] + 1)

# 1.9 кодируем event_type 0, 1, 2...
df['event_code'] = pd.factorize(df['event_type'])[0]

# 2. обучение модели 
# 2.1 собираем числовые признаки вместе
features = [
    'hour', 
    'ip_frequency', 
    'username_frequency', 
    'status_frequency', 
    'is_bash_command', 
    'event_code',
    'ip_cumulative_events',
    'ip_cumulative_failures',
    'ip_failure_rate'
]

# разделяем данные: обучаем модель только на чистых/нормальных
X_train = df[df['label'] == 'normal'][features]

# 2.2 масштабируем числа
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2.3 обучаем One-Class SVM 
# допускаем только 5% случайных ошибок/выбросов
model = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
model.fit(X_train_scaled)

# разделили на обучение, но предсказание делаем для всего df
X_all_scaled = scaler.transform(df[features].fillna(0))
df['predicted_anomaly'] = model.predict(X_all_scaled)

# SVM возвращает 1 - норма, -1 - аномалия. переводим в адекват
df['predicted_anomaly'] = df['predicted_anomaly'].map({1: 0, -1: 1})

print(pd.crosstab(df['label'], df['predicted_anomaly']))
