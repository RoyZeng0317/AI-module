import string

def read_input_character():
    # read the user input string, if isn't Enlgish character will be print the error
    ch = input('please enter a character: ')
    if ch not in string.ascii_letters:
        raise ValueError('enter character must need use English character!')
    return ch

try:
    ch = read_input_character()
    print('your enter the character is: ', ch)
except ValueError as e:
    print('enter error: ', str(e))