import math
import random


def get_k_fold(n, k):
    assert k > 1, "k must be greater than 1."
    assert n % k == 0, "n must be a multiple of k."

    numbers = list(range(n))  # A list of numbers from 0 to n

    # Randomly shuffled list of numbers
    random.shuffle(numbers)

    # Dividing the list of numbers into k parts, with each part containing n/k numbers
    length = n // k
    return [numbers[i:i + length] for i in range(0, n, length)]


def get_combination(lists, k):  # Selecting k elements from a set of n elements
    def backtrack(start, path):
        if len(path) == k:
            # Finding a combination
            result = []
            for p in path:
                result.extend(p)
            combinations.append(result)
            return
        for i in range(start, len(lists)):
            path.append(lists[i])
            backtrack(i + 1, path)
            path.pop()

    combinations = []
    backtrack(0, [])
    return combinations


def find_factors(n):
    factors = []
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if n // i != i:  # If i is not a square root, dividing n by i will yield another factor
                factors.append(n // i)
    return factors


def find_k_fold_plan(n, inn, min_k=4, max_k=10):
    plans = []
    factors = find_factors(n)
    print(f"Number of categories: {n}, expected number of combinations：{min_k}-{max_k}")
    for factor in factors:
        k = n // factor  # k groups，each length i factor
        # if inn % factor != 0:
        #     continue

        combs = []
        for i in range(1, k + 1):
            if i * factor == inn:
                combs.append((math.comb(k, i), (k, i)))

        for comb in combs:
            if min_k <= comb[0] <= max_k:
                k, i = comb[1]
                plans.append((k, i))
                print(f"k-fold scheme: selecting {i} out of {k}, number of combinations:{comb[0]}")
    return plans


if __name__ == '__main__':
    split_str = ' '

    class_num = int(input("Your class num: "))
    include_num = int(input("Your include class num: "))  # 已知类个数 include class
    plans = find_k_fold_plan(class_num, include_num)
    if len(plans) == 1:
        k, i = plans[0]
        print(f"Automatic selection scheme: selecting {i} out of {k}")
        fold_list = get_k_fold(class_num, k)
        comb_list = get_combination(fold_list, i)

        print(f"A total of {len(comb_list)} combinations: ")
        for c in comb_list:
            print(c)
        print(f"A total of {len(comb_list)} combinations (Space-separated): ")
        for c in comb_list:
            print(f"\"{split_str.join(map(str, c))}\"")
