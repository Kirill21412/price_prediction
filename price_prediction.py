import pandas as pd
from sklearn.linear_model import LinearRegression

# Пример исторических данных
data = {
    'day': [1, 2, 3, 4, 5],
    'price': [1000, 1100, 1150, 1200, 1250]
}

df = pd.DataFrame(data)
model = LinearRegression()

# Обучение модели
model.fit(df[['day']], df['price'])

# Прогнозирование цены на следующий день
next_day = 6
predicted_price = model.predict([[next_day]])

print(f"Predicted price on day {next_day}: {predicted_price[0]}")
