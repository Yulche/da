import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

# берем полученный ранее список работ
top_jobs = list(pd.read_csv('top_jobs.csv').name)

# берем список профессий и должностей каждой cv, исключаем NaN значения
works = pd.read_csv('works.csv')[['qualification', 'jobTitle']].dropna()

# создаем объект, выделяющий из русских слов основную неизменяемую часть (стему)
stemmer = nltk.stem.snowball.SnowballStemmer("russian")


# данная функция делит атрибут (работу) на слова, сверяет со списком работ и возвращает токены,
# присутствующие в обоих наборах данных (теги)
def get_tags(attribute):
    tokens = [stemmer.stem(wordpunkt) for wordpunkts in
              [nltk.wordpunct_tokenize(word) for word in nltk.word_tokenize(attribute)] for
              wordpunkt in wordpunkts]
    return set(tokens).intersection(top_jobs)


# для тепловой карты создаем счетчики выборки и DataFrame (таблицы отношений профессий и должностей)
init_sample = len(works)
final_sample = init_sample
heatmap = pd.DataFrame({job: [0] * 100 for job in top_jobs}, index=top_jobs)
for qlf, job in zip(works.qualification, works.jobTitle):
    # находим теги для профессии и должности cv
    qlf_tags = get_tags(qlf)
    job_tags = get_tags(job)
    # исключаем cv без тегов
    if len(qlf_tags) == 0 or len(job_tags) == 0:
        final_sample -= 1
        continue
    # согласно профессиям и должностям, которые занимает человек увеличиваем счетчики в тепловой карте
    for qt in qlf_tags:
        for jt in job_tags:
            heatmap[qt][jt] += 1


# функция для перечисляемой одномерной последовательности работ
# возвращает форматированную строку с топ-5 самых популярных работ
def get_top5(iterable):
    return ''.join([f'    {index + 1}. {item[1]} ({item[0]} резюме);\n' for index, item in
                    enumerate(sorted(zip(iterable, top_jobs), reverse=True)[:5])])


# функция для перечисляемой двумерной последовательности работ
# возвращает список из 5 самых популярных работ
def get_most_pop(data):
    return [job[1] for job in sorted([(sum(data[job]), job) for job in top_jobs], reverse=True)[:5]]


# вывод результатов анализа:
print(f'Начальная выборка без NaN значений состоит из {init_sample} резюме.\n'
      f'После отбора значений, подходящих под составленный "топ" профессий, в выборке осталось {final_sample} резюме.\n'
      f'Получившийся "топ" покрывает {round(final_sample / init_sample * 100, 2)}% выборки.\n'
      f'Статистика:\n'
      # респонденты, работающие по специальности - на главной диагонали матрицы-тепловой карты
      f'1. Не работают по профессии {100 - round(sum(heatmap[job][job] for job in top_jobs) / final_sample * 100, 2)}% респондентов.\n'
      f'2. Менеджерами чаще всего становятся респонденты с образованием:\n'
      f'{get_top5(heatmap.transpose()["менеджер"])}'
      f'3. Респонденты с образованием инженера чаще всего становятся:\n'
      f'{get_top5(heatmap["инженер"])}'
      f'4. Тепловая карта топ-5 профессий и топ-5 должностей.')

pop_qlf = get_most_pop(heatmap)
pop_job = get_most_pop(heatmap.transpose())
# Построение тепловой карты
sns.heatmap(heatmap[[*pop_qlf]].transpose()[[*pop_job]], cmap="icefire", annot=True, fmt="d")
plt.title('Тепловая карта топ-5 профессий и топ-5 должностей', fontsize=10)
plt.show()