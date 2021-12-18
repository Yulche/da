import pandas as pd
import nltk

# объединяем список должностей и работ
jobs = list(pd.read_csv('works.csv').qualification.dropna()) + list(pd.read_csv('works.csv').jobTitle.dropna())

# далее устанавливаем подбиблиотеку, которая необходима для разбиения предложений на слова
nltk.download('punkt')
# создаем объект, выделяющий из русских слов основную неизменяемую часть (называется стема)
stemmer = nltk.stem.snowball.SnowballStemmer("russian")

# создаем список окончаний названий работ
endings = ['ер', 'ир', 'ор', 'ар', 'ец', 'ик', 'ел', 'ист', 'ант', 'ог', 'ож', 'ач', 'ед', 'иц']

# создаем список слов, которые попадают в финальную выборку, но не являются названиями работ
# (их было проблематично отследить)
errors = ['начальник', 'руководител', 'заместител', 'завед', 'помощник', 'специалист', 'мастер', 'консультант',
          'технолог', 'работник', 'представител', 'делопроизводител', 'отдел', 'товар', 'свер', 'дел', 'категор',
          'издел', 'прибор', 'втор', 'кред', 'территор', 'сектор']

# создаем словарь для хранения неизменяемых частей и частот их появления
jobs_dict = {}

for job in jobs:
    # каждую работу делим на отдельные слова, получаем токены, исключаем токены меньше 3х букв
    tokens = [wordpunkt for wordpunkts in [nltk.wordpunct_tokenize(word) for word in nltk.word_tokenize(job)] for
              wordpunkt in wordpunkts if len(wordpunkt) > 3]
    for token in tokens:
        # для каждого токена выделяем неизменяемую часть
        stem_token = stemmer.stem(token)
        # исключаем неизменяемые части из списка "ошибок"
        if not any([stem_token.endswith(end) for end in endings]) or stem_token in errors:
            continue
        # в словарь добавляем неизменяемую часть, частоту ее появления и список токенов, из которых она была получена
        if stem_token not in jobs_dict:
            jobs_dict[stem_token] = [0, set()]
        jobs_dict[stem_token][0] += 1
        jobs_dict[stem_token][1].add(token)

# сортируем неизменяемые части по частоте появления и берем топ-100
top_jobs = [job for job in sorted(jobs_dict.items(), key=lambda i: i[1][0], reverse=True)][:100]

# преобразуем лист в словарь, создаем DataFrame и сохраняем в top_jobs9.csv
top_jobs = pd.DataFrame({'name': [job[0] for job in top_jobs],
                              'count': [job[1][0] for job in top_jobs],
                              'words': [', '.join(job[1][1]) for job in top_jobs]})
top_jobs.to_csv('top_jobs.csv', encoding='utf-8', index=False)