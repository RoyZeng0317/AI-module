# ch15-6 multiple error
def division(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        print("the division is 0")
    except TypeError:
        print("the error is use the string to calculate divison")

print(division(10, 2))
print(division(5, 0))
print(division('a', 'b'))
print(division(6, 3))