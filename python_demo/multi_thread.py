import time
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def cpu_bound(number):
    print(sum(i * i for i in range(number)))

# def calculate_sums(numbers):
#     for number in numbers:
#         cpu_bound(number)

# def calculate_sums(numbers):
#     threads = []
#     for number in numbers:
#         thread = Thread(cpu_bound(number))  # 实例化
#         thread.start()  # 启动线程
#         threads.append(thread)

#     for thread in threads:
#         thread.join()  # 等待线程完成，并关闭线程

# def calculate_sums(numbers):
#     pool = ThreadPoolExecutor(max_workers=4)  # 开了4个线程
#     results= list(pool.map(cpu_bound, numbers))

def calculate_sums(numbers):
    pool = ProcessPoolExecutor(max_workers=4)  # 开了4个线程
    results= list(pool.map(cpu_bound, numbers))

def main():
    start_time = time.perf_counter()
    numbers = [10000000 + x for x in range(4)]
    calculate_sums(numbers)
    end_time = time.perf_counter()
    print(f'Total time is {end_time-start_time:.4f} seconds')

if __name__ == '__main__':
    # main()

    aa = list((1, 2, 3))
    a = iter(aa)
    print(type(a))
