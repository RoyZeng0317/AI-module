# ch15-2
def division(x, y):
    # main division code
    try:    # maybe has the error
        return x / y
    except ZeroDivisionError:   # which part is error
        print("the division isn't 0.")
    except TypeError:   # remove the error and show the
        print("the type is error.")
# use comma to split it.
print(division(10, 2))
print(division(5, 0))
print(division('a', 'b'))
print(division(6, 3))