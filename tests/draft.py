from multiprocessing import Pool
from threading import Thread

def write(arr, start):
    print(start)
    for i in range(start, start + 2):
        arr.append(i)
    print(arr)

def main():
    arr = []
    pool = Pool(2)
    pool.starmap(write, ((arr, 1), (arr, 10)))
    print(arr)

if __name__ == '__main__':
    main()
