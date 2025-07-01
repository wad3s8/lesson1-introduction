import torch
import time
from prettytable import PrettyTable

print("Используемое устройство:")
if torch.cuda.is_available():
    print("CUDA (GPU)")
else:
    print("Только CPU. CUDA недоступна.")

cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda") if torch.cuda.is_available() else None

# 3.1 Подготовка данных
matrix_sizes = [
    (64, 1024, 1024),
    (128, 512, 512),
    (256, 256, 256),
]

matrices_cpu = {}
matrices_cuda = {}

for i, (d1, d2, d3) in enumerate(matrix_sizes):
    key_A = f"Mat_{i+1}_A"
    key_B = f"Mat_{i+1}_B"
    key_EA = f"Mat_{i+1}_EA"
    key_EB = f"Mat_{i+1}_EB"
    
    matrices_cpu[key_A] = torch.randn(d1, d2)
    matrices_cpu[key_B] = torch.randn(d2, d3)
    matrices_cpu[key_EA] = torch.randn(d1, d2)
    matrices_cpu[key_EB] = torch.randn(d1, d2)

    if cuda_device:
        matrices_cuda[key_A] = matrices_cpu[key_A].to(cuda_device)
        matrices_cuda[key_B] = matrices_cpu[key_B].to(cuda_device)
        matrices_cuda[key_EA] = matrices_cpu[key_EA].to(cuda_device)
        matrices_cuda[key_EB] = matrices_cpu[key_EB].to(cuda_device)

print("Матрицы созданы.")

# 3.2 Измерение времени

def measure_cpu(func, *args):
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    return (end - start) * 1000

def measure_cuda(func, *args):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

# 3.3 Сравнение операций
print("\n-- 3.3 Сравнение операций --")
table = PrettyTable()
table.field_names = ["Операция", "Размер", "CPU (мс)", "CUDA (мс)", "Ускорение"]

for i, (d1, d2, d3) in enumerate(matrix_sizes):
    key_A = f"Mat_{i+1}_A"
    key_B = f"Mat_{i+1}_B"
    key_EA = f"Mat_{i+1}_EA"
    key_EB = f"Mat_{i+1}_EB"

    operations = {
        f"Матр. умножение": (torch.matmul, matrices_cpu[key_A], matrices_cpu[key_B]),
        f"Сложение": (torch.add, matrices_cpu[key_EA], matrices_cpu[key_EB]),
        f"Умножение": (torch.mul, matrices_cpu[key_EA], matrices_cpu[key_EB]),
        f"Транспонирование": (torch.transpose, matrices_cpu[key_A], 0, 1),
        f"Суммирование": (torch.sum, matrices_cpu[key_EA])
    }

    for name, cpu_args in operations.items():
        cpu_time = measure_cpu(cpu_args[0], *cpu_args[1:])

        if cuda_device:
            cuda_args = tuple(matrices_cuda[k] if isinstance(k, torch.Tensor) else k for k in cpu_args[1:])
            cuda_time = measure_cuda(cpu_args[0], *cuda_args)
            speedup = cpu_time / cuda_time if cuda_time > 0 else float('inf')
            table.add_row([f"{name}", f"{d1}x{d2}", f"{cpu_time:.2f}", f"{cuda_time:.2f}", f"{speedup:.2f}x"])
        else:
            table.add_row([f"{name}", f"{d1}x{d2}", f"{cpu_time:.2f}", "N/A", "N/A"])

    table.add_row(["-"*20]*5)

print(table)

# 3.4 Анализ можно описать отдельно в README или отчёте