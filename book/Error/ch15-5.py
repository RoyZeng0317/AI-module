# ch15-5
def division(x, y):
    # main division code
    try:    # maybe has the error
        return x / y
    except Exception:   # which part is error
        print("all of error is happent.")
# use comma to split it.
print(division(10, 2))
print(division(5, 0))
print(division('a', 'b'))
print(division(6, 3))