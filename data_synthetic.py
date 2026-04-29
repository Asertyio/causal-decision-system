# data_synthetic.py
"""Синтетическая генерация карьерных данных.

Улучшения v2:
- Реалистичный self-selection через пропенсити-скор (п. 1)
- Временна́я динамика: прогнозы на 6 мес / 1 год / 2 года (п. 4)
- Interaction features: age×skills, experience×education (п. 6)
- current_salary / job_satisfaction / work_life_balance вынесены
  как baseline-значения и не попадают в ковариаты модели (п. 3)
"""
import numpy as np
import pandas as pd
from scipy.special import expit, softmax

# Полный список всех 30 действий, должен совпадать с causal_model.py и app.py
ALL_ACTIONS = [
    'Остаться на текущем месте',
    'Сменить работу (та же отрасль)',
    'Сменить работу с переездом в другой регион',
    'Сменить отрасль',
    'Получить второе высшее образование',
    'Пройти профессиональные курсы / сертификацию',
    'Начать подрабатывать / брать проекты',
    'Уйти во фриланс',
    'Открыть свой бизнес / стартап',
    'Выйти на пенсию / сократить занятость',
    'Взять академический отпуск',
    'Перейти на удалённую работу',
    'Перейти на частичную занятость',
    'Повысить квалификацию внутри компании',
    'Попросить повышения',
    'Попросить увеличения зарплаты',
    'Перейти в дочернюю компанию / филиал',
    'Релоцироваться в другую страну',
    'Сменить профессию полностью',
    'Пойти в декретный отпуск / отпуск по уходу за ребёнком',
    'Вернуться из декрета',
    'Начать инвестировать / трейдинг',
    'Заняться волонтёрством',
    'Вступить в профессиональное сообщество',
    'Начать вести блог / личный бренд',
    'Получить MBA',
    'Пройти стажировку',
    'Устроиться на вторую работу',
    'Участвовать в конференциях / нетворкинг',
    'Ничего не менять, продолжать как есть'
]


def _compute_propensity_scores(age, total_experience, education_years, job_level_num,
                                skills_count, job_satisfaction, current_salary,
                                is_moscow, gender) -> np.ndarray:
    """Вычисляет матрицу вероятностей выбора каждого из 30 действий
    в зависимости от профиля человека.

    Returns
    -------
    np.ndarray of shape (n, 30) — строки суммируются в 1.
    """
    n = len(age)
    # Base log-odds per action (will be adjusted by covariates)
    # Actions that are universally common get higher base odds
    base_log_odds = np.zeros((n, 30))

    # 0: Остаться на текущем месте — prefer older, low-satisfaction → reluctant
    base_log_odds[:, 0] = 1.0 + 0.03 * (age - 35) - 0.15 * job_satisfaction

    # 1: Сменить работу (та же отрасль) — prefer young + skilled
    base_log_odds[:, 1] = 0.5 + 0.02 * skills_count - 0.01 * (age - 25).clip(0)

    # 2: Сменить работу с переездом — prefer young, not already in Moscow
    base_log_odds[:, 2] = -0.5 + 0.03 * (age < 35) - 0.5 * is_moscow

    # 3: Сменить отрасль — prefer young + educated
    base_log_odds[:, 3] = -0.8 + 0.04 * (age < 32) + 0.05 * education_years

    # 4: Второе высшее образование — prefer low education, young
    base_log_odds[:, 4] = -1.0 + 0.1 * (education_years <= 2) + 0.02 * (age < 35)

    # 5: Курсы/сертификация — universally attractive, more so for mid-level
    base_log_odds[:, 5] = 0.8 + 0.03 * skills_count - 0.01 * (job_level_num - 3)**2

    # 6: Подрабатывать — prefer those with skills and lower salary
    base_log_odds[:, 6] = -0.3 + 0.03 * skills_count - 0.005 * current_salary

    # 7: Фриланс — prefer high skills, low satisfaction
    base_log_odds[:, 7] = -1.0 + 0.08 * skills_count - 0.2 * job_satisfaction

    # 8: Открыть бизнес — prefer young + experienced + skilled
    base_log_odds[:, 8] = -2.0 + 0.05 * (age < 40) + 0.04 * total_experience + 0.05 * skills_count

    # 9: Выйти на пенсию — very strong age signal
    base_log_odds[:, 9] = -5.0 + 0.12 * (age - 50).clip(0)

    # 10: Академический отпуск — young + high education
    base_log_odds[:, 10] = -2.0 + 0.05 * (age < 30) + 0.05 * education_years

    # 11: Удалённая работа — universally appealing but more for IT-age
    base_log_odds[:, 11] = 0.3 + 0.02 * skills_count

    # 12: Частичная занятость — older or low satisfaction
    base_log_odds[:, 12] = -1.5 + 0.03 * (age > 45) + 0.1 * (job_satisfaction < 5)

    # 13: Повысить квалификацию внутри — mid-level preference
    base_log_odds[:, 13] = 0.2 + 0.05 * (job_level_num == 2) + 0.02 * total_experience

    # 14: Попросить повышения — experienced, senior/lead
    base_log_odds[:, 14] = -0.5 + 0.05 * total_experience + 0.1 * (job_level_num >= 3)

    # 15: Попросить увеличения зарплаты — experienced
    base_log_odds[:, 15] = -0.2 + 0.04 * total_experience

    # 16: Перейти в дочернюю компанию — modest niche
    base_log_odds[:, 16] = -1.5 + 0.01 * total_experience

    # 17: Релоцироваться за рубеж — highly skilled, not in Moscow or seeking escape
    base_log_odds[:, 17] = -3.0 + 0.1 * skills_count + 0.05 * education_years - 0.3 * is_moscow

    # 18: Сменить профессию — young + educated
    base_log_odds[:, 18] = -1.5 + 0.04 * (age < 35) + 0.03 * education_years

    # 19: Декретный отпуск — gender signal (women, 25-40)
    is_woman = (gender == 'Женский').astype(float)
    base_log_odds[:, 19] = -4.0 + 2.5 * is_woman + 0.1 * ((age >= 25) & (age <= 40)).astype(float)

    # 20: Вернуться из декрета — women 28-45
    base_log_odds[:, 20] = -4.0 + 2.0 * is_woman + 0.1 * ((age >= 28) & (age <= 45)).astype(float)

    # 21: Инвестировать/трейдинг — higher salary, moscow
    base_log_odds[:, 21] = -1.0 + 0.008 * current_salary + 0.4 * is_moscow

    # 22: Волонтёрство — satisfaction-seeking
    base_log_odds[:, 22] = -2.0 + 0.1 * (job_satisfaction < 6)

    # 23: Профессиональное сообщество — mid+ level
    base_log_odds[:, 23] = -0.5 + 0.04 * (job_level_num >= 2) + 0.02 * skills_count

    # 24: Блог / личный бренд — skills + young
    base_log_odds[:, 24] = -1.5 + 0.05 * skills_count + 0.02 * (age < 40)

    # 25: MBA — experience + senior level
    base_log_odds[:, 25] = -2.5 + 0.05 * total_experience + 0.1 * (job_level_num >= 3) + 0.3 * is_moscow

    # 26: Стажировка — young + low experience
    base_log_odds[:, 26] = -1.0 + 0.1 * (age < 28) + 0.1 * (total_experience < 3)

    # 27: Вторая работа — motivated by money, lower salary
    base_log_odds[:, 27] = -1.0 - 0.005 * current_salary + 0.02 * skills_count

    # 28: Конференции / нетворкинг — senior, skilled
    base_log_odds[:, 28] = -0.3 + 0.04 * (job_level_num >= 3) + 0.02 * skills_count

    # 29: Ничего не менять — inertia, high satisfaction
    base_log_odds[:, 29] = 0.5 + 0.1 * job_satisfaction - 0.01 * (age - 35)**2 / 10

    # Softmax across actions to get probabilities
    probs = softmax(base_log_odds, axis=1)
    return probs


def generate_career_data(n=10000, random_state=42):
    np.random.seed(random_state)

    age = np.random.randint(22, 65, size=n)
    gender = np.random.choice(['Мужской', 'Женский'], size=n, p=[0.52, 0.48])

    regions = [
        'Алтайский край', 'Амурская область', 'Архангельская область', 'Астраханская область',
        'Белгородская область', 'Брянская область', 'Владимирская область', 'Волгоградская область',
        'Вологодская область', 'Воронежская область', 'Еврейская АО', 'Забайкальский край',
        'Ивановская область', 'Иркутская область', 'Кабардино-Балкарская Республика', 'Калининградская область',
        'Калужская область', 'Камчатский край', 'Карачаево-Черкесская Республика', 'Кемеровская область',
        'Кировская область', 'Костромская область', 'Краснодарский край', 'Красноярский край',
        'Курганская область', 'Курская область', 'Ленинградская область', 'Липецкая область',
        'Магаданская область', 'Москва', 'Московская область', 'Мурманская область', 'Ненецкий АО',
        'Нижегородская область', 'Новгородская область', 'Новосибирская область', 'Омская область',
        'Оренбургская область', 'Орловская область', 'Пензенская область', 'Пермский край',
        'Приморский край', 'Псковская область', 'Республика Адыгея', 'Республика Алтай',
        'Республика Башкортостан', 'Республика Бурятия', 'Республика Дагестан', 'Республика Ингушетия',
        'Республика Калмыкия', 'Республика Карелия', 'Республика Коми', 'Республика Крым',
        'Республика Марий Эл', 'Республика Мордовия', 'Республика Саха (Якутия)', 'Республика Северная Осетия-Алания',
        'Республика Татарстан', 'Республика Тыва', 'Республика Хакасия', 'Ростовская область',
        'Рязанская область', 'Самарская область', 'Санкт-Петербург', 'Саратовская область',
        'Сахалинская область', 'Свердловская область', 'Севастополь', 'Смоленская область',
        'Ставропольский край', 'Тамбовская область', 'Тверская область', 'Томская область',
        'Тульская область', 'Тюменская область', 'Удмуртская Республика', 'Ульяновская область',
        'Хабаровский край', 'Ханты-Мансийский АО', 'Челябинская область', 'Чеченская Республика',
        'Чувашская Республика', 'Чукотский АО', 'Ямало-Ненецкий АО', 'Ярославская область'
    ]
    region_probs = np.ones(len(regions))
    region_probs[regions.index('Москва')] = 10
    region_probs[regions.index('Санкт-Петербург')] = 6
    region_probs[regions.index('Московская область')] = 5
    region_probs = region_probs / region_probs.sum()
    region = np.random.choice(regions, size=n, p=region_probs)

    edu_years_choices = [0, 2, 4, 6, 9, 1]
    edu_probs = [0.03, 0.15, 0.45, 0.25, 0.07, 0.05]
    education_years = np.random.choice(edu_years_choices, size=n, p=edu_probs)
    has_master = (education_years >= 6).astype(int)
    has_phd = (education_years >= 9).astype(int)
    has_certificate = np.random.binomial(1, 0.35, size=n)

    total_experience = np.maximum(0, age - 18 - education_years + np.random.normal(0, 2, size=n)).astype(int)
    total_experience = np.clip(total_experience, 0, 60)
    industry_experience = np.random.binomial(total_experience, 0.7)
    current_job_tenure = np.random.binomial(industry_experience, 0.3)
    num_previous_jobs = np.maximum(0, np.random.poisson(total_experience / 3.5)).astype(int)
    num_previous_jobs = np.clip(num_previous_jobs, 0, 30)

    industries = [
        'IT и разработка ПО', 'Финансы и банки', 'Нефтегазовая отрасль', 'Металлургия', 'Энергетика',
        'Телекоммуникации', 'Ритейл', 'E-commerce', 'Строительство', 'Транспорт и логистика',
        'Авиастроение', 'Автомобилестроение', 'Машиностроение', 'Химическая промышленность',
        'Фармацевтика', 'Медицина и здравоохранение', 'Сельское хозяйство', 'Пищевая промышленность',
        'Образование', 'Наука и исследования', 'Консалтинг', 'Аудит', 'Юриспруденция',
        'Государственная служба', 'НКО и благотворительность', 'Маркетинг и реклама', 'PR',
        'Дизайн', 'Медиа и журналистика', 'Кино и телевидение', 'Игровая индустрия',
        'Туризм и гостиничный бизнес', 'Ресторанный бизнес', 'Спорт и фитнес', 'Красота и уход',
        'Лесная промышленность', 'Рыболовство', 'Добыча полезных ископаемых', 'Страхование',
        'Инвестиции', 'Венчурный капитал', 'Лизинг', 'Факторинг', 'Микрофинансирование',
        'Брокерские услуги', 'Операции с недвижимостью', 'Геодезия и картография',
        'Водоснабжение и водоотведение', 'Управление отходами', 'Лёгкая промышленность',
        'Ювелирное дело', 'Полиграфия', 'Издательское дело', 'Архитектура', 'Градостроительство',
        'Ландшафтный дизайн', 'Ветеринария', 'Зообизнес', 'Охрана и безопасность',
        'Клининговые услуги', 'Ремонт и обслуживание техники', 'Прокат и аренда',
        'Социальное обслуживание', 'Психология и коучинг', 'Астрология и эзотерика', 'Фриланс',
        'Самозанятость', 'Стартап', 'Искусство и культура', 'Музейное дело', 'Библиотечное дело',
        'Религиозные организации', 'Политика', 'Международные отношения', 'Таможенное дело',
        'Авиаперевозки', 'Железнодорожный транспорт', 'Морской транспорт', 'Речной транспорт',
        'Метрополитен', 'Космическая отрасль', 'Оборонная промышленность', 'Атомная энергетика',
        'Горнодобывающая отрасль', 'Целлюлозно-бумажная отрасль', 'Мебельное производство',
        'Производство стройматериалов', 'Производство электроники', 'Приборостроение',
        'Робототехника', 'Биотехнологии', 'Нанотехнологии', 'Экология и природопользование',
        'Водные ресурсы', 'Лесное хозяйство', 'Другие отрасли'
    ]
    ind_probs = np.ones(len(industries))
    popular = ['IT и разработка ПО', 'Финансы и банки', 'Ритейл', 'Строительство', 'Образование']
    for ind in popular:
        if ind in industries:
            ind_probs[industries.index(ind)] = 3
    ind_probs = ind_probs / ind_probs.sum()
    current_industry = np.random.choice(industries, size=n, p=ind_probs)

    job_levels = ['Junior', 'Middle', 'Senior', 'Lead', 'Manager']
    level_probs = [0.2, 0.4, 0.25, 0.1, 0.05]
    job_level = np.random.choice(job_levels, size=n, p=level_probs)
    level_map = {'Junior': 1, 'Middle': 2, 'Senior': 3, 'Lead': 4, 'Manager': 5}
    job_level_num = np.array([level_map[x] for x in job_level])

    skills_count = np.clip(np.random.poisson(5, size=n) + job_level_num * 2, 1, 20).astype(int)

    base_salary = 35
    region_mult = np.ones(n)
    region_mult[(region == 'Москва') | (region == 'Санкт-Петербург')] = 2.2
    region_mult[region == 'Московская область'] = 1.6
    region_mult[region == 'Ленинградская область'] = 1.5
    region_mult += np.random.uniform(-0.15, 0.25, n)
    region_mult = np.clip(region_mult, 0.9, 2.8)

    level_mult = np.array([0.7, 1.0, 1.5, 2.0, 2.5])[job_level_num - 1]

    industry_mult = np.ones(n)
    high_pay = ['IT и разработка ПО', 'Нефтегазовая отрасль', 'Финансы и банки', 'Добыча полезных ископаемых']
    low_pay = ['Образование', 'НКО и благотворительность', 'Социальное обслуживание', 'Библиотечное дело']
    for i, ind in enumerate(current_industry):
        if ind in high_pay:
            industry_mult[i] = 1.7
        elif ind in low_pay:
            industry_mult[i] = 0.75

    exp_bonus = 2.0 * np.sqrt(total_experience)
    edu_bonus = education_years * 1.8
    skill_bonus = skills_count * 1.0

    current_salary = (base_salary * region_mult * level_mult * industry_mult +
                      exp_bonus + edu_bonus + skill_bonus +
                      np.random.normal(0, 15, size=n))
    current_salary = np.maximum(20, current_salary)

    job_satisfaction = np.clip(
        5.5 + (current_salary / 100) - (current_job_tenure / 4) +
        np.random.normal(0, 1.8, n), 1, 10
    ).round(1)
    work_life_balance = np.clip(
        6.5 - (job_level_num * 0.7) + (current_salary / 120) +
        np.random.normal(0, 1.5, n), 1, 10
    ).round(1)

    # ── УЛУЧШЕНИЕ 1: Реалистичный self-selection через пропенсити-скор ──
    is_moscow = (region == 'Москва') | (region == 'Санкт-Петербург')
    prop_scores = _compute_propensity_scores(
        age, total_experience, education_years, job_level_num,
        skills_count, job_satisfaction, current_salary, is_moscow.astype(float), gender
    )
    treatment = np.array([
        np.random.choice(ALL_ACTIONS, p=prop_scores[i])
        for i in range(n)
    ])

    # ── УЛУЧШЕНИЕ 4: Временна́я динамика (6 мес, 1 год, 2 года) ──
    # Эффекты масштабируются по горизонту через factor: action_effect_6m < 1y < 2y
    # Закодированы как доли от полного 2-летнего эффекта
    TIME_FACTORS = {'6m': 0.35, '1y': 0.65, '2y': 1.0}

    Y_salary = {h: np.zeros(n) for h in TIME_FACTORS}
    Y_sat = {h: np.zeros(n) for h in TIME_FACTORS}
    Y_prom = {h: np.zeros(n) for h in TIME_FACTORS}
    Y_wlb = {h: np.zeros(n) for h in TIME_FACTORS}

    for i in range(n):
        sal = current_salary[i]
        sat = job_satisfaction[i]
        wlb = work_life_balance[i]
        age_i = age[i]
        exp_i = total_experience[i]
        edu_i = education_years[i]
        level_i = job_level_num[i]
        skills_i = skills_count[i]
        cert_i = has_certificate[i]
        reg_i = region[i]
        is_moscow_i = reg_i in ['Москва', 'Санкт-Петербург']

        # Base trajectory: small natural growth regardless of action
        sal_base = sal * (1 + 0.025 * np.sqrt(exp_i) + 0.005 * edu_i)

        t = treatment[i]

        # ── Heterogeneous treatment effects (effect depends on who you are) ──
        if t == 'Остаться на текущем месте':
            stagnation = -0.005 * current_job_tenure[i]
            sal_effect = 1.03 + stagnation
            sat_effect = -0.3 - 0.05 * current_job_tenure[i]
            prom_boost = 0.02 * level_i
            wlb_effect = 0.15
            # Быстро сказывается: нет задержки
            time_sal = {'6m': 0.5, '1y': 0.75, '2y': 1.0}

        elif t == 'Сменить работу (та же отрасль)':
            mult = (1.22 if age_i < 35 else 1.10) + 0.008 * skills_i
            sal_effect = mult
            sat_effect = 1.2 if age_i < 40 else 0.4
            prom_boost = 0.10 + 0.02 * (level_i <= 2)
            wlb_effect = -0.5 - 0.1 * (level_i >= 4)
            time_sal = {'6m': 0.6, '1y': 0.8, '2y': 1.0}

        elif t == 'Сменить работу с переездом в другой регион':
            mult = 1.45 if not is_moscow_i else 1.10
            sal_effect = mult
            sat_effect = 1.5 if age_i < 35 else 0.4
            prom_boost = 0.15
            wlb_effect = -1.2
            time_sal = {'6m': 0.5, '1y': 0.75, '2y': 1.0}

        elif t == 'Сменить отрасль':
            if age_i < 32 and edu_i >= 4:
                sal_effect = 1.45
                sat_effect = 2.2
                prom_boost = 0.12
            elif age_i < 45:
                sal_effect = 1.05
                sat_effect = 0.5
                prom_boost = 0.08
            else:
                sal_effect = 0.85
                sat_effect = -1.2
                prom_boost = 0.05
            wlb_effect = -0.8
            time_sal = {'6m': 0.3, '1y': 0.6, '2y': 1.0}  # slow ramp

        elif t == 'Получить второе высшее образование':
            sal_effect = 1.18 + 0.03 * (edu_i <= 2)
            sat_effect = 0.9 + 0.2 * (age_i < 35)
            prom_boost = 0.28
            wlb_effect = -0.8
            time_sal = {'6m': 0.1, '1y': 0.4, '2y': 1.0}  # degree takes time

        elif t == 'Пройти профессиональные курсы / сертификацию':
            sal_effect = 1.10 + 0.015 * skills_i + 0.02 * cert_i
            sat_effect = 0.6
            prom_boost = 0.20
            wlb_effect = -0.3
            time_sal = {'6m': 0.5, '1y': 0.85, '2y': 1.0}

        elif t == 'Начать подрабатывать / брать проекты':
            sal_effect = 1.12 + 0.01 * skills_i
            sat_effect = 0.3 if sat < 6 else -0.3
            prom_boost = 0.04
            wlb_effect = -1.5
            time_sal = {'6m': 0.7, '1y': 0.9, '2y': 1.0}

        elif t == 'Уйти во фриланс':
            if skills_i >= 10:
                sal_effect = np.random.choice([1.6, 1.1, 0.8], p=[0.45, 0.35, 0.20])
            else:
                sal_effect = np.random.choice([1.2, 0.85, 0.65], p=[0.25, 0.40, 0.35])
            sat_effect = 1.5 if sal_effect > 1.0 else -1.0
            prom_boost = 0.0
            wlb_effect = 1.0 if sal_effect > 1.0 else -0.5
            time_sal = {'6m': 0.4, '1y': 0.7, '2y': 1.0}

        elif t == 'Открыть свой бизнес / стартап':
            if age_i < 40 and exp_i > 3 and skills_i >= 7:
                sal_effect = np.random.choice([2.2, 1.0, 0.55], p=[0.35, 0.30, 0.35])
            elif exp_i > 3:
                sal_effect = np.random.choice([1.5, 0.7], p=[0.3, 0.7])
            else:
                sal_effect = 0.65
            sat_effect = 2.5 if sal_effect >= 1.5 else (-1.8 if sal_effect < 0.8 else 0.5)
            prom_boost = 0.0
            wlb_effect = -2.8
            time_sal = {'6m': 0.2, '1y': 0.5, '2y': 1.0}  # business takes time

        elif t == 'Выйти на пенсию / сократить занятость':
            sal_effect = 0.42
            sat_effect = 1.5 + 0.05 * max(0, age_i - 55)
            prom_boost = 0.0
            wlb_effect = 3.0
            time_sal = {'6m': 0.8, '1y': 0.9, '2y': 1.0}

        elif t == 'Взять академический отпуск':
            sal_effect = 0.0
            sat_effect = 1.0
            prom_boost = 0.05
            wlb_effect = 2.0
            time_sal = {'6m': 0.0, '1y': 0.0, '2y': 1.0}  # income returns after

        elif t == 'Перейти на удалённую работу':
            sal_effect = 1.05
            sat_effect = 0.9 if age_i < 45 else 0.1
            prom_boost = 0.04
            wlb_effect = 2.0
            time_sal = {'6m': 0.8, '1y': 0.95, '2y': 1.0}

        elif t == 'Перейти на частичную занятость':
            sal_effect = 0.65
            sat_effect = 0.8
            prom_boost = 0.01
            wlb_effect = 2.5
            time_sal = {'6m': 0.9, '1y': 1.0, '2y': 1.0}

        elif t == 'Повысить квалификацию внутри компании':
            sal_effect = 1.10 + 0.01 * current_job_tenure[i]
            sat_effect = 0.7
            prom_boost = 0.25
            wlb_effect = 0.0
            time_sal = {'6m': 0.4, '1y': 0.75, '2y': 1.0}

        elif t == 'Попросить повышения':
            success_p = 0.55 if (exp_i > 3 and level_i <= 3) else 0.30
            success = np.random.binomial(1, success_p)
            sal_effect = (1.22 if success else 1.02)
            sat_effect = 1.2 if success else -0.5
            prom_boost = 0.35 if success else 0.05
            wlb_effect = -0.2
            time_sal = {'6m': 0.9, '1y': 1.0, '2y': 1.0}

        elif t == 'Попросить увеличения зарплаты':
            success_p = 0.60 if exp_i > 2 else 0.35
            success = np.random.binomial(1, success_p)
            sal_effect = (1.18 if success else 1.01)
            sat_effect = 0.8 if success else -0.3
            prom_boost = 0.05
            wlb_effect = -0.1
            time_sal = {'6m': 0.95, '1y': 1.0, '2y': 1.0}

        elif t == 'Перейти в дочернюю компанию / филиал':
            sal_effect = 1.12
            sat_effect = 0.5
            prom_boost = 0.18
            wlb_effect = -0.4
            time_sal = {'6m': 0.6, '1y': 0.85, '2y': 1.0}

        elif t == 'Релоцироваться в другую страну':
            sal_effect = 2.2 if skills_i > 10 else (1.6 if skills_i > 6 else 1.1)
            sat_effect = 1.5 if sal_effect > 1.5 else 0.3
            prom_boost = 0.20
            wlb_effect = -1.5 + 0.05 * skills_i
            time_sal = {'6m': 0.5, '1y': 0.75, '2y': 1.0}

        elif t == 'Сменить профессию полностью':
            if age_i < 35 and edu_i >= 4:
                sal_effect = 1.30
                sat_effect = 2.0
            else:
                sal_effect = 0.80
                sat_effect = -0.5
            prom_boost = 0.08
            wlb_effect = -0.5
            time_sal = {'6m': 0.2, '1y': 0.5, '2y': 1.0}

        elif t == 'Пойти в декретный отпуск / отпуск по уходу за ребёнком':
            sal_effect = 0.50
            sat_effect = 1.0
            prom_boost = 0.0
            wlb_effect = 1.5
            time_sal = {'6m': 0.5, '1y': 0.5, '2y': 1.0}

        elif t == 'Вернуться из декрета':
            sal_effect = 0.88 + 0.02 * edu_i
            sat_effect = 0.5
            prom_boost = 0.08
            wlb_effect = -0.5
            time_sal = {'6m': 0.7, '1y': 0.9, '2y': 1.0}

        elif t == 'Начать инвестировать / трейдинг':
            invest_gain = np.random.choice([1.0, 0.0, -0.5], p=[0.4, 0.4, 0.2])
            sal_effect = 1.0 + 0.08 * invest_gain
            sat_effect = 0.3 * invest_gain
            prom_boost = 0.01
            wlb_effect = -0.3
            time_sal = {'6m': 0.5, '1y': 0.8, '2y': 1.0}

        elif t == 'Заняться волонтёрством':
            sal_effect = 1.0
            sat_effect = 1.2
            prom_boost = 0.05
            wlb_effect = 0.5
            time_sal = {'6m': 0.6, '1y': 0.8, '2y': 1.0}

        elif t == 'Вступить в профессиональное сообщество':
            sal_effect = 1.05 + 0.01 * skills_i
            sat_effect = 0.6
            prom_boost = 0.12
            wlb_effect = 0.0
            time_sal = {'6m': 0.4, '1y': 0.7, '2y': 1.0}

        elif t == 'Начать вести блог / личный бренд':
            sal_effect = 1.08 + 0.015 * skills_i
            sat_effect = 0.8
            prom_boost = 0.10
            wlb_effect = -0.4
            time_sal = {'6m': 0.2, '1y': 0.6, '2y': 1.0}

        elif t == 'Получить MBA':
            sal_effect = (1.40 if level_i >= 3 else 1.18) + 0.02 * is_moscow_i
            sat_effect = 0.7
            prom_boost = 0.42
            wlb_effect = -1.2
            time_sal = {'6m': 0.1, '1y': 0.5, '2y': 1.0}  # MBA takes 1-2 years

        elif t == 'Пройти стажировку':
            sal_effect = 0.75 if exp_i > 5 else 1.05
            sat_effect = 0.5 if exp_i <= 5 else -0.5
            prom_boost = 0.15 if exp_i <= 5 else 0.05
            wlb_effect = -0.3
            time_sal = {'6m': 0.6, '1y': 0.85, '2y': 1.0}

        elif t == 'Устроиться на вторую работу':
            sal_effect = 1.20 + 0.005 * skills_i
            sat_effect = -0.8
            prom_boost = 0.03
            wlb_effect = -2.2
            time_sal = {'6m': 0.8, '1y': 0.95, '2y': 1.0}

        elif t == 'Участвовать в конференциях / нетворкинг':
            sal_effect = 1.06 + 0.01 * skills_i
            sat_effect = 0.5
            prom_boost = 0.14
            wlb_effect = -0.2
            time_sal = {'6m': 0.3, '1y': 0.65, '2y': 1.0}

        else:  # 'Ничего не менять, продолжать как есть'
            sal_effect = 1.02
            sat_effect = -0.15
            prom_boost = 0.01
            wlb_effect = 0.05
            time_sal = {'6m': 0.6, '1y': 0.8, '2y': 1.0}

        # Compose outcomes for each time horizon
        for h, tf in TIME_FACTORS.items():
            # Salary: time-dependent ramp
            sal_tf = time_sal.get(h, tf)
            sal_effect_h = 1.0 + (sal_effect - 1.0) * sal_tf
            sal_pred = sal_base * sal_effect_h + np.random.normal(0, sal * 0.05)

            # Sat/WLB/promo ramp by common factor
            sat_pred = sat + sat_effect * tf + np.random.normal(0, 0.7)
            wlb_pred = wlb + wlb_effect * tf + np.random.normal(0, 0.6)
            prom_prob = expit(-2.5 + 0.08 * exp_i + 0.15 * edu_i + prom_boost * 5 * tf + 0.04 * level_i)
            promoted = np.random.binomial(1, prom_prob)

            Y_salary[h][i] = np.maximum(15, sal_pred)
            Y_sat[h][i] = np.clip(sat_pred, 1, 10).round(1)
            Y_wlb[h][i] = np.clip(wlb_pred, 1, 10).round(1)
            Y_prom[h][i] = promoted

    # ── УЛУЧШЕНИЕ 6: Interaction features ──
    age_x_skills = age * skills_count / 100.0
    exp_x_edu = total_experience * education_years / 10.0

    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'region': region,
        'education_years': education_years,
        'has_master': has_master,
        'has_phd': has_phd,
        'has_certificate': has_certificate,
        'total_experience': total_experience,
        'industry_experience': industry_experience,
        'current_job_tenure': current_job_tenure,
        'num_previous_jobs': num_previous_jobs,
        'current_industry': current_industry,
        'job_level': job_level,
        'skills_count': skills_count,
        # Baseline values (kept for outcome generation, but EXCLUDED from model features)
        'current_salary': current_salary,
        'job_satisfaction': job_satisfaction,
        'work_life_balance': work_life_balance,
        'treatment': treatment,
        # Primary (2-year) outcomes
        'salary_2y': Y_salary['2y'],
        'satisfaction_2y': Y_sat['2y'],
        'promoted_2y': Y_prom['2y'],
        'wlb_2y': Y_wlb['2y'],
        # 1-year outcomes
        'salary_1y': Y_salary['1y'],
        'satisfaction_1y': Y_sat['1y'],
        'promoted_1y': Y_prom['1y'],
        'wlb_1y': Y_wlb['1y'],
        # 6-month outcomes
        'salary_6m': Y_salary['6m'],
        'satisfaction_6m': Y_sat['6m'],
        'promoted_6m': Y_prom['6m'],
        'wlb_6m': Y_wlb['6m'],
    })

    df['experience_gap'] = df['total_experience'] - df['industry_experience']
    df['job_stability'] = df['current_job_tenure'] / (df['total_experience'] + 1)
    # Interaction features
    df['age_x_skills'] = age_x_skills
    df['exp_x_edu'] = exp_x_edu

    # Guarantee all 30 actions appear at least once so the model trains on all of them
    present_actions = set(df['treatment'].unique())
    missing_actions = set(ALL_ACTIONS) - present_actions
    if missing_actions:
        extra_rows = []
        for act in missing_actions:
            sample_row = df.sample(1).iloc[0].copy()
            sample_row['treatment'] = act
            extra_rows.append(sample_row)
        df_extra = pd.DataFrame(extra_rows)
        df = pd.concat([df, df_extra], ignore_index=True)

    return df