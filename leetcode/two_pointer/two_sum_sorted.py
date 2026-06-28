# tags: two-pointer, array
# Past solved solution, kept only as an analogy for "find a pair" problems.


def two_sum_sorted(numbers, target):
    lo, hi = 0, len(numbers) - 1
    while lo < hi:
        total = numbers[lo] + numbers[hi]
        if total == target:
            return [lo, hi]
        if total < target:
            lo += 1
        else:
            hi -= 1
    return [-1, -1]
