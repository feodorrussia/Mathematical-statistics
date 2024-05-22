import numpy as np
from sklearn.linear_model import LinearRegression

# Создаем фиктивную мультимодальную выборку
np.random.seed(0)
x1 = np.random.normal(-1, 1, size=500)  # первая мода
x2 = np.random.normal(2, 1, size=500)  # вторая мода
x = np.concatenate([x1, x2]).reshape(-1, 1)
a = np.array([3, 2])
# Создаем зависимую переменную с шумом
y = a * x.flatten() + np.random.normal(loc=0, scale=2, size=len(x))

# Создаем модель линейной регрессии
model = LinearRegression()

# Обучаем модель
model.fit(x, y)

# Получаем оценки коэффициентов
coef = model.coef_

# Распечатываем коэффициенты
print(f'Коэффициент: {coef[0]}')
