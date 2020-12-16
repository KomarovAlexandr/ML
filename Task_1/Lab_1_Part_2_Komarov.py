# -*- coding: utf-8 -*-
from collections import Counter
import re
import os

PATH = 'C:/Users/al.kom/YandexDisk-raptor.al.com@yandex.ru/Machine Learning/Lab_1/'
SPAM = 'spam'
HAM = 'ham'
os.mkdir(PATH + SPAM)
os.mkdir(PATH + HAM)

SMSSpamCollection = open(PATH + 'SpamSMS/SMSSpamCollection.txt', 'r', encoding="utf8")

init_lenght = len(SMSSpamCollection.readline())  # для инициализации массивов с длинами
SMSSpamCollection.close()
SMSSpamCollection = open(PATH + 'SpamSMS/SMSSpamCollection.txt', 'r', encoding="utf8")
# [number_of_message, number_of_spam_message, number_of_ham_message]
number_of_message = [0, 0, 0]  # массив кол-в сообщений в категории
# [max_message_length, min_message_length, av_message_length]
message_length = [init_lenght, init_lenght, init_lenght]  # Масив длинн для всех сообщений
# [max_spam_message_length, min_spam_message_length, av_spam_message_length]
spam_message_length = [init_lenght, init_lenght, init_lenght]  # Массив длинн для спам-сообщений
# [max_ham_message_length, min_ham_message_length, av_ham_message_length]
ham_message_length = [init_lenght, init_lenght, init_lenght]  # Массив длинн для обычных сообщений

data_set = ''  # данные для поиска популярных слов во всех сообщениях
data_set_spam = ''  # данные для поиска популярных слов в спаме
data_set_ham = ''  # данные для поиска популярных слов в обычных сообщениях
mas_most_occur_word = []  # список саммых популярных слов во всех сообщениях
mas_most_occur_word_spam = []  # список саммых популярных слов в спаме
mas_most_occur_word_ham = []  # список саммых популярных слов в обычных сообщениях

# Кол-во сообщений в которых встречается хотя бы одно из 20 популярных слов
number_containing_one_popular_words_out_of_20 = [0,  # слова из спама в спаме
                                                 0,  # слова из спама в обычных
                                                 0,  # слова из обычных в спаме
                                                 0]  # слова из обычных в обычных
# Кол-во сообщений в которых встречается хотя бы одно из 10 популярных слов
number_containing_one_popular_words_out_of_10 = [0,  # слова из спама в спаме
                                                 0,  # слова из спама в обычных
                                                 0,  # слова из обычных в спаме
                                                 0]  # слова из обычных в обычных


def length_comparison(string, mas):  # Функция сравнения параметров сообщения с данными в массиве
    if len(string) > mas[0]:  # сравнение с известной максимальной длинной
        mas[0] = len(string)
    if len(string) < mas[1]:  # сравнение с известной минимальной длинной
        mas[1] = len(string)
    mas[2] = mas[2] + len(string)  # накопление значения для средней длинны
    return mas


def common_words(data):  # функция для поиска популярных слов в категории
    data = data.lower()  # понижаем регистр, т.к. он не имеет значения
    split_it = data.split()  # находим 20 самых частых слов в категории
    counter = Counter(split_it)
    most_occur = counter.most_common(20)
    return most_occur


def search_for_popular_words_in_message(data, popular_word_mas, num):
    number = 0
    while data.find('\n') > -1:  # откусываем по строке
        position = data.find('\n')  # находим конец строки
        string = data[0:position].split()  # забираем строку
        data = data[(position + 1):len(data)]  # уменьшаем размер входного наборо строк
        find_flag = False
        for i in range(num):  # выполняем поиск каждого популярного слова в строке
            for word in string:
                if word == popular_word_mas[i][0]:
                    number = number + 1
                    find_flag = True  # Как только нашли совпадение заканиваем поиск для этой строки
                    break
            if find_flag:
                find_flag = False
                break
    return number


def find_longest_words(data):  # Функция поиска 20 самых длинных слов
    dataset = data.lower()
    words = re.split(':|\'|;|&|\|<|>|!|,|\(|\)|\+|\?|\*|\.|\n|\d|\s|/|~|=|-|@', dataset)  # делим через заданные символы
    sorted_words = sorted(words, key=len, reverse=True)  # сортируем с ключем-длинной
    number_of_longest_words = 0
    i = 0
    while number_of_longest_words <= 20:
        if sorted_words[i] == sorted_words[i+1]:
            del sorted_words[i]
            i = i - 1
        number_of_longest_words = number_of_longest_words + 1
        i = i + 1
    return sorted_words[0:20]


for line in SMSSpamCollection:
    new_line = line
    new_line = new_line[4:len(new_line)]
    new_line = new_line.lstrip()
    data_set = data_set + new_line
    message_length = length_comparison(new_line, message_length)
    if line.find(HAM, 0, 4) > -1:
        file = open(PATH + HAM + '/' + str(number_of_message[0]) + '.txt', 'w', encoding="utf8")
        file.write(new_line)
        file.close()
        data_set_ham = data_set_ham + '\n' + new_line
        ham_message_length = length_comparison(new_line, ham_message_length)
        number_of_message[2] = number_of_message[2] + 1
    if line.find(SPAM, 0, 4) > -1:
        file = open(PATH + SPAM + '/' + str(number_of_message[0]) + '.txt', 'w', encoding="utf8")
        file.write(new_line)
        file.close()
        data_set_spam = data_set_spam + '\n' + new_line
        spam_message_length = length_comparison(new_line, spam_message_length)
        number_of_message[1] = number_of_message[1] + 1
    number_of_message[0] = number_of_message[0] + 1

SMSSpamCollection.close()

# находим среднее значение длины слов в категории
message_length[2] = int(message_length[2] / number_of_message[0])
spam_message_length[2] = int(spam_message_length[2] / number_of_message[1])
ham_message_length[2] = int(ham_message_length[2] / number_of_message[2])

# находим 20 самых популярных слов в категории
mas_most_occur_word = common_words(data_set)
mas_most_occur_word_spam = common_words(data_set_spam)
mas_most_occur_word_ham = common_words(data_set_ham)

# Кол-во сообщений в которых встречается хотя бы одно из 20 популярных слов
number_containing_one_popular_words_out_of_20[0] = search_for_popular_words_in_message(data_set_spam, mas_most_occur_word_spam, 20)  # слова из спама в спаме
number_containing_one_popular_words_out_of_20[1] = search_for_popular_words_in_message(data_set_ham, mas_most_occur_word_spam, 20)  # слова из спама в обычных
number_containing_one_popular_words_out_of_20[2] = search_for_popular_words_in_message(data_set_spam, mas_most_occur_word_ham, 20)  # слова из обычных в спаме
number_containing_one_popular_words_out_of_20[3] = search_for_popular_words_in_message(data_set_ham, mas_most_occur_word_ham, 20)  # слова из обычных в обычных

# Кол-во сообщений в которых встречается хотя бы одно из 10 популярных слов
number_containing_one_popular_words_out_of_10[0] = search_for_popular_words_in_message(data_set_spam, mas_most_occur_word_spam, 10)  # слова из спама в спаме
number_containing_one_popular_words_out_of_10[1] = search_for_popular_words_in_message(data_set_ham, mas_most_occur_word_spam, 10)  # слова из спама в обычных
number_containing_one_popular_words_out_of_10[2] = search_for_popular_words_in_message(data_set_spam, mas_most_occur_word_ham, 10)  # слова из обычных в спаме
number_containing_one_popular_words_out_of_10[3] = search_for_popular_words_in_message(data_set_ham, mas_most_occur_word_ham, 10)  # слова из обычных в обычных

# 20 самых длинных слов из всех сообщений
longest_words = find_longest_words(data_set)


""" ВЫВОД ВСЕГО-ВСЕГО В OUTPUT-ФАЙЛ """
Output_file = open(PATH + 'Output.txt', 'w', encoding="utf8")
Output_file.write("Кол-во сообщений всего: " + str(number_of_message[0]) + '\n'
                  "Кол-во спам-сообщений: " + str(number_of_message[1]) + '\n'
                  "Кол-во обычных сообщений: " + str(number_of_message[2]) + 2*'\n')

Output_file.write("Максимальная длина сообщения из всех: " + str(message_length[0]) + '\n'
                  "Минимальная длина сообщения из всех: " + str(message_length[1]) + '\n'
                  "Средняя длина сообщения из всех: " + str(message_length[2]) + 2*'\n')

Output_file.write("Максимальная длина сообщения из спама: " + str(spam_message_length[0]) + '\n'
                  "Минимальная длина сообщения из спама: " + str(spam_message_length[1]) + '\n'
                  "Средняя длина сообщения из спама: " + str(spam_message_length[2]) + 2*'\n')

Output_file.write("Максимальная длина сообщения из обычных сообщений: " + str(ham_message_length[0]) + '\n'
                  "Минимальная длина сообщения из обычных сообщений: " + str(ham_message_length[1]) + '\n'
                  "Средняя длина сообщения из обычных сообщений: " + str(ham_message_length[2]) + 2*'\n')

Output_file.write("20 самых популярных слов из всех сообщений" + '\n' + '   ')
for i in mas_most_occur_word:
    Output_file.write(str(i) + ', ')
Output_file.write(2*'\n')

Output_file.write("20 самых популярных слов в спаме " + '\n' + '   ')
for i in mas_most_occur_word_spam:
    Output_file.write(str(i) + ', ')
Output_file.write(2*'\n')

Output_file.write("20 самых популярных слов в обычных сообщениях " + '\n' + '   ')
for i in mas_most_occur_word_ham:
    Output_file.write(str(i) + ', ')
Output_file.write(2*'\n')

Output_file.write("Кол-во сообщений в которых встречается хотя бы одно из 20 популярных слов" + '\n' + '   ')
Output_file.write("20 слов из спама в спам-сообщениях: " + str(number_containing_one_popular_words_out_of_20[0]) + '\n' + '   ')
Output_file.write("20 слов из спама в обычных сообщениях: " + str(number_containing_one_popular_words_out_of_20[1]) + '\n' + '   ')
Output_file.write("20 слов из обычных в спам-сообщениях: " + str(number_containing_one_popular_words_out_of_20[2]) + '\n' + '   ')
Output_file.write("20 слов из обычных в обычных сообщениях: " + str(number_containing_one_popular_words_out_of_20[3]) + '\n' + '   ')
Output_file.write('\n')

Output_file.write("Кол-во сообщений в которых встречается хотя бы одно из 10 популярных слов" + '\n' + '   ')
Output_file.write("10 слов из спама в спам-сообщениях: " + str(number_containing_one_popular_words_out_of_10[0]) + '\n' + '   ')
Output_file.write("10 слов из спама в обычных сообщениях: " + str(number_containing_one_popular_words_out_of_10[1]) + '\n' + '   ')
Output_file.write("10 слов из обычных в спам-сообщениях: " + str(number_containing_one_popular_words_out_of_10[2]) + '\n' + '   ')
Output_file.write("10 слов из обычных в обычных сообщениях: " + str(number_containing_one_popular_words_out_of_10[3]) + '\n' + '   ')
Output_file.write('\n')

Output_file.write("20 самых длинных слов во всех сообщениях" + '\n')
for i in longest_words:
    Output_file.write(i + ', \n')

Output_file.close()
