import torch

# 1.1 Создание тензоров
first_tensor = torch.rand(3,4)
second_tensor = torch.zeros(2,3,4)
third_tensor = torch.ones(5,5)
sequence = torch.arange(16)
fourth_tensor = torch.reshape(sequence, (4, 4))

print(f'Тензор размером 3x4, заполненный случайными числами от 0 до 1: {first_tensor}')
print(f'Тензор размером 2x3x4, заполненный нулями: {second_tensor}')
print(f'Тензор размером 5x5, заполненный единицами: {third_tensor}')
print(f'Тензор размером 4x4 с числами от 0 до 15: {fourth_tensor}')

# 1.2 Операции с тензорами
a = torch.randint(0, 10, (3, 4))
b = torch.randint(0, 10, (4, 3))

a_transp = a.T
print(f'Транспонирование тензора A: {a_transp}')

matmul_task = a @ b
print(f'Матричное умножение: {matmul_task}')

print(f'Поэлементное умножение: {a * b.T}')

sum_a = a.sum()
print(f'Сумма всех элеентов тензора A: {sum_a}')

# 1.3 Индексация и срезы
indexing_cutting_tensor = torch.randint(0, 5, (5, 5, 5))

print(f'Весь tensor: {indexing_cutting_tensor}')
print(f'Первая строка: {indexing_cutting_tensor[:1, :1]}')
print(f'Последний стоблец: {indexing_cutting_tensor[:, :, -1:]}')
print(f'Подматрицу размером 2x2 из центра тензора: {indexing_cutting_tensor[2, 1:3, 1:3]}')
print(f'Все элементы с четными индексами: {indexing_cutting_tensor[::2, ::2, ::2]}')

# 1.4 Работа с формами
tensor24 = torch.arange(24)

print(f'2x12: {tensor24.reshape(2,12)}')
print(f'3x8: {tensor24.reshape(3,8)}')
print(f'4x6: {tensor24.reshape(4,6)}')
print(f'2x3x4: {tensor24.reshape(2,3,4)}')
print(f'2x2x2x3: {tensor24.reshape(2,2,2,3)}')




