# ch15-7
def division(x, y):
    try:
        return x / y
    except (ZeroDivisionError, TypeError):
        print("the division is 0 or the error is use the string to calculate divison")

print(division(10, 2))
print(division(5, 0))
print(division('a', 'b'))
print(division(6, 3))