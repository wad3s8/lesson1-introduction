import torch


#2.1 Простые вычисления с градиентами
# Создаем тензоры с requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

# Вычисляем функцию f = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2 * x * y * z

# Вызываем backward для вычисления градиентов
f.backward()

# Выводим градиенты
print(f"df/dx = {x.grad}")
print(f"df/dy = {y.grad}")
print(f"df/dz = {z.grad}")

#Проверка
print("Аналитическая проверка")
print(f"df/dx = {2*x + 2*y*z}")
print(f"df/dy = {2*y + 2*x*z}")
print(f"df/dz = {2*z + 2*y*x}")


# 2.2 Градиент функции потерь
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

# Параметры модели с requires_grad=True
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Линейная модель: y_pred = w * x + b
y_pred = w * x + b

# MSE: среднеквадратичная ошибка
mse = ((y_pred - y_true) ** 2).mean()

# Вычисление градиентов
mse.backward()

# Градиенты по w и b
print(f"Градиент по w: {w.grad}")
print(f"Градиент по b: {b.grad}")


# 2.3 Цепное правило
x = torch.tensor(2.0, requires_grad=True)

f = torch.sin(x**2 + 1)
f.backward()

print(f"Градиент через .backward(): {x.grad}")

x = torch.tensor(2.0, requires_grad=True)
f = torch.sin(x**2 + 1)
grad = torch.autograd.grad(f, x)[0]
print(f"Градиент через torch.autograd.grad: {grad}")


