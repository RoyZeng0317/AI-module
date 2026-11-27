# ch15-3
fn = 'data15_3.txt'

try:
    with open(fn) as file_Obj:
        data = file_Obj.read()
except FileNotFoundError:
    print(f"Can't find the {fn} this file.")
else:
    print(data)