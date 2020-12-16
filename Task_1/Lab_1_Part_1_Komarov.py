def sum_mult(list_a, list_b):
    if len(list_a) != len(list_b):  # если длины не равны, None
        print("Не равны длины списков")
        return None

    list_of_mult = []  # массив произведений
    list_of_sum = []  # массив сумм
    for i in range(len(list_a)):
        try:
            list_of_mult.append(list_a[i] * list_b[i])
            list_of_sum.append(list_a[i] + list_b[i])
        except:
            print("В списке есть символы")
            return None
    return [list_of_sum, list_of_mult]


def palindrome(inp_str):
    if type(inp_str) != str:  # провекра на строку
        print("Не строка!")
        return None

    my_string = inp_str

    while my_string.find(' ') > -1:  # цикл для удаления всех пробелов из строки
        i = my_string.find(' ')
        my_string = my_string[0:i] + my_string[(i + 1):len(my_string)]

    if len(my_string) < 2:  # проверка на длинну
        print("Длина строки меньше 2!")
        return None

    print(inp_str)
    my_string = my_string.lower()

    for i in range(len(my_string)):  # перебираем символы и сравниваем их
        if my_string[i] != my_string[-1 - i]:
            print("Не полиндром")
            return False
    print("Полиндром")
    return True


def max_digit_sum(stop_str, max_numbers):
    list_of_number = []

    while True:
        message = input("Введите число или стоп слово ")
        if message == stop_str:
            break
        print(message)
        try:
            list_of_number.append(float(message))
        except:
            print("Введено не число")
        if len(list_of_number) >= max_numbers:
            break

    print(list_of_number)
    max_number_index = 0
    max_sum = 0

    for i in range(len(list_of_number)):
        sum_of_num = 0
        number = abs(list_of_number[i])
        if str(number).find('.') > -1:
            number = number * pow(10, len(str(number)) - str(number).find('.'))
        while number != 0:
            sum_of_num = sum_of_num + number % 10
            number = number // 10
        if i == 0:
            max_number_index = i
            max_sum = sum_of_num
        if sum_of_num > max_sum:
            max_number_index = i
            max_sum = sum_of_num

    return list_of_number[max_number_index] - max_sum


def goose_weight(breed, sex, age_days):
    table = {'H': {'m': {1: 99, 10: 267, 20: 756, 30: 1390, 40: 1800, 50: 2310, 60: 3510, 90: 5320, 120: 6000, 160: 6420},
                    'f': {1: 106, 10: 254, 20: 698, 30: 1300, 40: 1730, 50: 2170, 60: 3280, 90: 4470, 120: 5210, 160: 5680} },
            'A': {'m': {1: 99, 10: 394, 20: 965, 30: 1820, 40: 2513, 50: 3440, 60: 4025, 90: 4760, 120: 5250, 160: 5625},
                    'f': {1: 95, 10: 395, 20: 977, 30: 1871, 40: 2520, 50: 3230, 60: 3840, 90: 4540, 120: 4630, 160: 5006} },
            'U': {'m': {1: 100, 10: 292, 20: 791, 30: 1410, 40: 1970, 50: 2520, 60: 3420, 90: 4350, 120: 4700, 160: 5110} ,
                    'f': {1: 98, 10: 310, 20: 715, 30: 1230, 40: 1920, 50: 2330, 60: 3190, 90: 3780, 120: 4140, 160: 4530} },
            'K': {1: 102, 10: 250, 20: 700, 30: 1360, 40: 1600, 50: 2600, 60: 3900, 90: 4700, 120: 5100, 160: 5400} }

    try:
        if breed == 'K':
            return table[breed][age_days]
        else:
            return table[breed][sex][age_days]
    except:
        print("Не верные параметры")
        return None
    
