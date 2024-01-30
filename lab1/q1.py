def pairs_with_sum(arr, target_sum):
    count = 0
    check = set()

    for num in arr:
        complement = target_sum - num
        if complement in check:
            count += 1
        check.add(num)

    return count

given_list = [2, 7, 4, 1, 3, 6]
target_sum = 10

result = pairs_with_sum(given_list, target_sum)
print(f"Number of pairs with sum equal to {target_sum} is  {result}")
