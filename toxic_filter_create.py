# -*- coding: utf-8 -*-
import time
from deap import base, creator, tools
import random

# Ф-я нормализации текста.
def normalize(text):
    txt = text.replace(' ', '')
    txt = txt.replace('ё', 'е')
    txt = txt.replace('й', 'и')
    return txt

def load_file(file):
    with open(file, encoding='utf-8-sig') as input_file:
        lst = []
        for line in input_file:
            str = normalize(line.strip())
            if len(str) > 1: # убираем строки длиной менее 2х
                lst.append(str)
    return lst

good_words = load_file('good_words.txt')
# print(good_words)
bad_words = load_file('bad_words.txt')
# print(bad_words)

# Словарь со статистикой паттернов
patterns = {}

# Добавление в словарь и ведение статистики
def pattern_stat(p):
    # p = p.strip()
    if p in patterns:
        patterns[p] += 1
    else:
        patterns.update({p: 1})

# Создание паттерна (подстрока)
def create_pattern(text, letter_count):
    for pos in range(0, len(text) - letter_count):
        pattern_stat(text[pos: pos + letter_count])


min_letter_count = 2
max_letter_count = 7

for letter_count in range(min_letter_count, max_letter_count):
    for w in bad_words:
        create_pattern(normalize(w), letter_count)

# for letter_count in range(min_letter_count, max_letter_count):
#     for w in good_words:
#         create_pattern(normalize(w), letter_count)

patterns = sorted(patterns.items(),
                  key=lambda patterns: patterns[1], reverse=True)
print('Кол-во паттернов:', len(patterns))

# Переводит бинарный вектор в текстовый массив - фильтр
def decode(individual):
    lst = []
    for pos in range(0, len(individual)):
        if (individual[pos] == 1):
            lst.append(patterns[pos][0])
    return lst

# % строк исследуемого текста (набора фраз bad|good), которые "чистятся" фильтром
def percent_in_text(word_set, filter):
    count = 0
    for w in word_set:
        for p in filter:
            if p in w:
                count += 1
                break
    return count / len(word_set)

# ГА
start_time = time.time()

# Целевая функция
def eval_func(individual):
    txt = decode(individual)
    return [percent_in_text(bad_words, txt), percent_in_text(good_words, txt), sum(map(len, txt))]

# Создание toolbox
def create_toolbox(num_bits):
    creator.create("FitnessCompound", base.Fitness, weights=(1.0, -1.0, -0.6))
    creator.create("Individual", list, fitness=creator.FitnessCompound)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, num_bits)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


population_size = 100
num_generations = 1000
# Вероятности скрещивания и мутации
probab_crossing, probab_mutating = 0.5, 0.3

# Размер гена равен кол-ву паттернов (one hot)
num_bits = len(patterns)
toolbox = create_toolbox(num_bits)
random.seed(7)
population = toolbox.population(n=population_size)

print('--- Старт ГА ---')
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

for g in range(num_generations):
    start_gen_time = time.time()
    # Выбор следующего поколения
    offspring = toolbox.select(population, len(population))
    # Клонирование выбранных экземпляров
    offspring = list(map(toolbox.clone, offspring))
    # Применяем скрещивание
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < probab_crossing:
            toolbox.mate(child1, child2)
            # Удаляем значение ЦФ наследников
            del child1.fitness.values
            del child2.fitness.values
    # Применяем мутацию
    for mutant in offspring:
        if random.random() < probab_mutating:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    # Оценка популяции
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Статистика текущего поколения
    print("Поколение %i (%s сек)" % (g, (time.time() - start_gen_time)))
    population.sort(key=lambda x: x.fitness, reverse=True)
    print('- Оценено', len(invalid_ind), 'экземпляров')
    print('- Лучший экземпляр: ', population[0].fitness.values)
    print()

    # Заменяем популяцию на следующее поколение
    population[:] = offspring

print('--- Стоп ГА ---')
print("Всего %s сек\n" % (time.time() - start_time))

# Итог работы алгоритма
best_ind = tools.selBest(population, 1)[0]
print('Лучший экземпляр:\n', best_ind)
print('Кол-во едениц:\n', sum(best_ind), 'из', len(best_ind))

toxic_filter = decode(best_ind)
# Сортируем фильтр по убыванию длины строки, так как более короткие паттерны чаще встречаются в "хорошем тексте",
# а длинные точнее определяют мат. 
# Надо добавить еще сортировку по частоте.
toxic_filter.sort(key=len, reverse=True)

print('Лучший фильтр:\n', toxic_filter)
print('Общая длина строки:\n', sum(map(len, toxic_filter)))
print('Точность по bad_words:\n', percent_in_text(bad_words, toxic_filter))
print('Точность по good_words:\n', percent_in_text(good_words, toxic_filter))