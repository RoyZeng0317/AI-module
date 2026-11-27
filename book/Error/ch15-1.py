# ch15-1
def division(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        print("the division isn't 0")
print(division(10, 2))
print(division(5, 0))
print(division(6, 3)) 