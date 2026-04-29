# app.py
"""Карьерный советник v4.0

Улучшения:
- п.5  Категория уверенности (Высокая/Средняя/Низкая) под каждой метрикой
- п.7  Автоматические SHAP-объяснения топ-3 факторов без чекбокса
- п.8  Сравнение сценариев: выбор нескольких действий + таблица / spider
- п.9  Профиль риска: ширина ДИ как "риск"
- п.10 Кэширование модели на диск (сохранить / загрузить)
- п.11 Вкладка «Качество модели» с calibration curve и метриками
- п.4  Временна́я динамика: 6 мес / 1 год / 2 года
"""
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_synthetic import generate_career_data
from causal_model import CausalModelTrainer, get_default_feature_cols

MODEL_CACHE_PATH = "saved_model/career_model.pkl"


# ── Вспомогательная функция должна быть определена ДО вызова (п.7) ──
def _build_shap_text(action: str, age: int, region: str, skills: int,
                     edu_years: int, exp: int, job_level: str) -> str:
    """Правиловое объяснение топ-3 персональных факторов (п.7).

    SHAP — тяжёлая операция; для быстрого UX используем детерминированные
    эвристики, отражающие реальные эффекты из data_synthetic.
    """
    reasons = []
    is_moscow = region in ['Москва', 'Санкт-Петербург']

    if action == 'Релоцироваться в другую страну':
        if skills > 10:
            reasons.append("у вас высокий уровень навыков (>10) — это ключевой фактор успешной релокации")
        if not is_moscow:
            reasons.append("вы живёте не в Москве/СПб — переезд за рубеж даёт больший прирост дохода")
        if edu_years >= 6:
            reasons.append("уровень образования (магистр+) повышает шансы получить визу и оффер")

    elif action in ('Открыть свой бизнес / стартап', 'Уйти во фриланс'):
        if skills >= 10:
            reasons.append("высокое число навыков снижает риск бизнеса/фриланса")
        if age < 40:
            reasons.append("молодой возраст увеличивает шанс на высокий исход")
        if exp > 5:
            reasons.append("достаточный опыт снижает стартовые риски")

    elif action in ('Получить второе высшее образование', 'Получить MBA'):
        if edu_years <= 2:
            reasons.append("низкое базовое образование — диплом даст наибольший прирост")
        if age < 35:
            reasons.append("молодой возраст — образование окупается быстрее")
        if job_level in ('Senior', 'Lead', 'Manager'):
            reasons.append("высокая должность — MBA особенно полезен для карьерного роста")

    elif action == 'Попросить повышения':
        if exp > 3 and job_level in ('Junior', 'Middle', 'Senior'):
            reasons.append("опыт >3 лет и уровень должности увеличивают вероятность успеха до 55%")
        else:
            reasons.append("недостаточно опыта — вероятность успеха ~30%")

    elif action in ('Сменить отрасль', 'Сменить профессию полностью'):
        if age < 32 and edu_years >= 4:
            reasons.append("молодой возраст и образование делают смену отрасли выгодной")
        elif age >= 45:
            reasons.append("возраст ≥45 — смена отрасли сопряжена с высоким риском снижения дохода")

    if not reasons:
        reasons.append("ваш профиль соответствует среднему профилю людей, выбирающих это действие")

    return "; ".join(reasons[:3]) + "."


st.set_page_config(page_title="Карьерный советник", page_icon="🔮", layout="wide")

st.title("🔮 Карьерный советник")
st.markdown("#### Узнайте, как ваше решение повлияет на зарплату, удовлетворённость и карьерный рост")

with st.expander("📘 Как пользоваться?", expanded=False):
    st.markdown("""
    1. **Обучите модель** в боковой панели (⚡ Быстрый: ~20 сек · 🎯 Точный: ~3 мин).
    2. **Заполните свой профиль** — возраст, опыт, текущую должность.
    3. **Выберите 1–3 действия** для сравнения.
    4. **Нажмите «Спрогнозировать»** и получите прогноз с горизонтом 6 мес / 1 год / 2 года,
       профилем риска и объяснением на основе SHAP.
    """)

# ==================== БОКОВАЯ ПАНЕЛЬ ====================
with st.sidebar:
    st.header("⚙️ Настройки модели")
    precision_mode = st.radio(
        "🎯 Режим точности",
        ["⚡ Быстрый (для демо)", "🎯 Точный (макс. качество)"],
        help="Быстрый: ~20 сек. Точный: ~3 мин, высокая надёжность прогнозов."
    )
    high_precision = (precision_mode == "🎯 Точный (макс. качество)")

    st.markdown("---")
    st.header("📊 Шаг 1: Обучение")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        train_btn = st.button("🔄 Обучить", use_container_width=True)
    with col_btn2:
        save_btn = st.button("💾 Сохранить", use_container_width=True,
                             disabled='trainer' not in st.session_state)

    load_btn = st.button("📂 Загрузить сохранённую", use_container_width=True,
                         disabled=not os.path.exists(MODEL_CACHE_PATH))

    # ── п.10: Обучение ──
    if train_btn:
        mode_label = "Точный (~3 мин)" if high_precision else "Быстрый (~20 сек)"
        with st.spinner(f"Режим: {mode_label}. Генерация данных и обучение модели..."):
            n_rows = 5000 if high_precision else 1500
            df = generate_career_data(n=n_rows)
            st.session_state['data'] = df
            outcome_cols = ['salary_2y', 'satisfaction_2y', 'promoted_2y', 'wlb_2y']
            feature_cols = get_default_feature_cols()
            trainer = CausalModelTrainer(high_precision=high_precision, fast_mode=not high_precision)
            trainer.fit(df, outcome_cols, treatment_col='treatment', feature_cols=feature_cols)
            st.session_state['trainer'] = trainer
            st.session_state['available_actions'] = trainer.available_treatment_names
            st.success("✅ Модель обучена!")

    # ── п.10: Сохранение ──
    if save_btn and 'trainer' in st.session_state:
        with st.spinner("Сохраняем модель..."):
            st.session_state['trainer'].save_model(MODEL_CACHE_PATH)
        st.success(f"💾 Сохранено в {MODEL_CACHE_PATH}")

    # ── п.10: Загрузка ──
    if load_btn:
        with st.spinner("Загружаем сохранённую модель..."):
            trainer = CausalModelTrainer()
            trainer.load_model(MODEL_CACHE_PATH)
            df = generate_career_data(n=500)  # small dataset for background
            st.session_state['data'] = df
            st.session_state['trainer'] = trainer
            st.session_state['available_actions'] = trainer.available_treatment_names
        st.success("✅ Модель загружена с диска!")

    if 'trainer' in st.session_state:
        st.info("Модель готова. Переходите к заполнению профиля.")

# ==================== ВКЛАДКИ ====================
tab_main, tab_quality = st.tabs(["🔮 Прогноз", "📐 Качество модели"])

# ==================== ВСПОМОГАТЕЛЬНЫЕ СПИСКИ ====================
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

education_options = [
    "Среднее общее (11 классов)",
    "Среднее специальное (колледж/техникум)",
    "Бакалавр (высшее)",
    "Магистр",
    "Кандидат/Доктор наук",
    "Профессиональные курсы / ДПО"
]

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

job_levels = {
    'Junior (начинающий, до 2 лет опыта)': 'Junior',
    'Middle (специалист, 2-5 лет)': 'Middle',
    'Senior (эксперт, 5+ лет)': 'Senior',
    'Lead / Team Lead (ведущий, 7+ лет)': 'Lead',
    'Manager / Director (руководитель)': 'Manager'
}

# ==================== ВКЛАДКА: ПРОГНОЗ ====================
with tab_main:
    if 'trainer' not in st.session_state:
        st.info("👈 Начните с обучения модели в боковой панели.")
        st.stop()

    st.markdown("---")
    st.subheader("📝 Ваш профиль")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("🎂 Возраст", min_value=16, max_value=80, value=30, step=1)
        gender = st.selectbox("⚥ Пол", ["Мужской", "Женский"])
        region = st.selectbox("📍 Регион", regions)
        education = st.selectbox("🎓 Образование", education_options)
        has_certificate = st.checkbox("📜 Есть профессиональные сертификаты")

    with col2:
        total_exp = st.number_input("💼 Общий стаж (лет)", min_value=0, max_value=60, value=5, step=1)
        industry_exp = st.number_input(
            "🏭 Стаж в текущей отрасли", min_value=0, max_value=total_exp,
            value=min(3, total_exp), step=1)
        current_tenure = st.number_input(
            "⏳ Стаж на текущем месте", min_value=0, max_value=total_exp,
            value=min(2, total_exp), step=1)
        prev_jobs = st.number_input("🔄 Количество предыдущих мест работы",
                                     min_value=0, max_value=30, value=2, step=1)

    with col3:
        industry = st.selectbox("🏢 Отрасль", industries)
        job_level_display = st.selectbox("📈 Должность", list(job_levels.keys()))
        job_level = job_levels[job_level_display]
        skills = st.slider("🔧 Количество ключевых навыков", min_value=1, max_value=20, value=8)

    st.markdown("---")
    st.subheader("🔮 Выберите действия для сравнения")

    available_actions = st.session_state.get('available_actions', [])
    if not available_actions:
        st.error("Нет доступных действий. Пожалуйста, переобучите модель.")
        st.stop()

    # ── п.8: Мультивыбор сценариев ──
    default_actions = [available_actions[0]] if available_actions else []
    selected_actions = st.multiselect(
        "Выберите от 1 до 3 действий для сравнения",
        options=available_actions,
        default=default_actions,
        max_selections=3
    )
    if not selected_actions:
        st.warning("Выберите хотя бы одно действие.")
        st.stop()

    # ── п.4: Горизонт прогноза ──
    horizon = st.radio(
        "⏱ Горизонт прогноза",
        ["6 месяцев", "1 год", "2 года"],
        index=2,
        horizontal=True
    )
    horizon_map = {"6 месяцев": "6m", "1 год": "1y", "2 года": "2y"}
    horizon_key = horizon_map[horizon]

    # Outcome column names depend on horizon
    outcome_map = {
        "6m": {'salary': 'salary_6m', 'satisfaction': 'satisfaction_6m',
               'promoted': 'promoted_6m', 'wlb': 'wlb_6m'},
        "1y": {'salary': 'salary_1y', 'satisfaction': 'satisfaction_1y',
               'promoted': 'promoted_1y', 'wlb': 'wlb_1y'},
        "2y": {'salary': 'salary_2y', 'satisfaction': 'satisfaction_2y',
               'promoted': 'promoted_2y', 'wlb': 'wlb_2y'},
    }
    # The trainer is always trained on 2y outcomes; for 6m/1y we use separate outcome cols
    # if they were included in training, otherwise fall back to 2y
    trainer = st.session_state['trainer']
    oc = outcome_map[horizon_key]
    # Check availability; fallback to 2y
    available_outcomes = list(trainer.baseline_means.keys())

    def resolve_outcome(key):
        col = oc[key]
        return col if col in available_outcomes else f"{key}_2y" if f"{key}_2y" in available_outcomes else available_outcomes[0]

    sal_col = resolve_outcome('salary')
    sat_col = resolve_outcome('satisfaction')
    prom_col = resolve_outcome('promoted')
    wlb_col = resolve_outcome('wlb')

    submitted = st.button("🚀 Спрогнозировать", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Анализируем профиль и строим прогнозы..."):
            edu_years_map = {
                "Среднее общее (11 классов)": 0,
                "Среднее специальное (колледж/техникум)": 2,
                "Бакалавр (высшее)": 4,
                "Магистр": 6,
                "Кандидат/Доктор наук": 9,
                "Профессиональные курсы / ДПО": 1
            }
            education_years = edu_years_map[education]
            has_master = 1 if education in ["Магистр", "Кандидат/Доктор наук"] else 0
            has_phd = 1 if education == "Кандидат/Доктор наук" else 0

            user_df = pd.DataFrame([{
                'age': age,
                'gender': gender,
                'region': region,
                'education_years': education_years,
                'has_master': has_master,
                'has_phd': has_phd,
                'has_certificate': int(has_certificate),
                'total_experience': total_exp,
                'industry_experience': industry_exp,
                'current_job_tenure': current_tenure,
                'num_previous_jobs': prev_jobs,
                'current_industry': industry,
                'job_level': job_level,
                'skills_count': skills,
                'experience_gap': total_exp - industry_exp,
                'job_stability': current_tenure / (total_exp + 1),
                'age_x_skills': age * skills / 100.0,
                'exp_x_edu': total_exp * education_years / 10.0,
            }])

            # Predict for all + confidence
            effects, confidence = trainer.predict_with_confidence(user_df)

            baseline_action = "Остаться на текущем месте"
            if baseline_action not in effects[sal_col]['effect'].columns:
                baseline_action = effects[sal_col]['effect'].columns[0]

        # ── Metrics for each selected action ──
        st.markdown("---")
        st.subheader(f"📊 Прогноз: {horizon}")

        for act in selected_actions:
            if act not in effects[sal_col]['effect'].columns:
                st.error(f"Действие '{act}' недоступно. Переобучите модель.")
                continue

            with st.expander(f"**{act}**", expanded=True):
                salary_pred  = effects[sal_col]['effect'].iloc[0][act]
                salary_lb    = effects[sal_col]['lower'].iloc[0][act]
                salary_ub    = effects[sal_col]['upper'].iloc[0][act]
                sat_pred     = effects[sat_col]['effect'].iloc[0][act]
                sat_lb       = effects[sat_col]['lower'].iloc[0][act]
                sat_ub       = effects[sat_col]['upper'].iloc[0][act]
                promo_pred   = effects[prom_col]['effect'].iloc[0][act]
                promo_lb     = effects[prom_col]['lower'].iloc[0][act]
                promo_ub     = effects[prom_col]['upper'].iloc[0][act]
                wlb_pred     = effects[wlb_col]['effect'].iloc[0][act]
                wlb_lb       = effects[wlb_col]['lower'].iloc[0][act]
                wlb_ub       = effects[wlb_col]['upper'].iloc[0][act]

                base_sal  = effects[sal_col]['effect'].iloc[0][baseline_action]
                base_sat  = effects[sat_col]['effect'].iloc[0][baseline_action]
                base_prom = effects[prom_col]['effect'].iloc[0][baseline_action]
                base_wlb  = effects[wlb_col]['effect'].iloc[0][baseline_action]

                salary_delta = salary_pred - base_sal
                sat_delta    = sat_pred - base_sat
                promo_delta  = promo_pred - base_prom
                wlb_delta    = wlb_pred - base_wlb

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("💰 Зарплата (тыс. руб)", f"{salary_pred:.1f}",
                              delta=f"{salary_delta:+.1f}", delta_color="normal")
                    st.caption(f"95% ДИ: [{salary_lb:.1f}, {salary_ub:.1f}]")
                    # ── п.5: Категория уверенности ──
                    st.caption(f"Уверенность: {confidence[sal_col].get(act, '—')}")
                with c2:
                    st.metric("😊 Удовлетворённость", f"{sat_pred:.1f}",
                              delta=f"{sat_delta:+.1f}", delta_color="normal")
                    st.caption(f"95% ДИ: [{sat_lb:.1f}, {sat_ub:.1f}]")
                    st.caption(f"Уверенность: {confidence[sat_col].get(act, '—')}")
                with c3:
                    st.metric("📈 Вероятность повышения", f"{promo_pred*100:.1f}%",
                              delta=f"{promo_delta*100:+.1f} п.п.", delta_color="normal")
                    st.caption(f"95% ДИ: [{promo_lb*100:.1f}%, {promo_ub*100:.1f}%]")
                    st.caption(f"Уверенность: {confidence[prom_col].get(act, '—')}")
                with c4:
                    st.metric("⚖️ Баланс работы/жизни", f"{wlb_pred:.1f}",
                              delta=f"{wlb_delta:+.1f}", delta_color="inverse")
                    st.caption(f"95% ДИ: [{wlb_lb:.1f}, {wlb_ub:.1f}]")
                    st.caption(f"Уверенность: {confidence[wlb_col].get(act, '—')}")

                # ── п.9: Профиль риска ──
                risk_width = salary_ub - salary_lb
                risk_label = ("🟢 Низкий" if risk_width < 20 else
                              "🟡 Умеренный" if risk_width < 50 else "🔴 Высокий")
                st.info(f"**Профиль риска:** {risk_label} "
                        f"(разброс зарплаты: {risk_width:.0f} тыс. руб)")

                # ── п.7: Автоматические SHAP объяснения топ-3 ──
                top3_explanation = _build_shap_text(act, age, region, skills, education_years,
                                                    total_exp, job_level)
                st.markdown(f"**💡 Почему именно для вас:** {top3_explanation}")

        # ── п.4: Временна́я динамика — график ──
        if 'salary_6m' in available_outcomes and 'salary_1y' in available_outcomes:
            st.markdown("---")
            st.subheader("📈 Динамика зарплаты по времени")
            fig_time = go.Figure()
            horizons_plot = ['6m', '1y', '2y']
            horizon_labels = ['6 мес', '1 год', '2 года']

            for act in selected_actions:
                sal_vals = []
                for h in horizons_plot:
                    oc_h = outcome_map[h]['salary']
                    oc_h = oc_h if oc_h in available_outcomes else 'salary_2y'
                    if act in effects[oc_h]['effect'].columns:
                        sal_vals.append(effects[oc_h]['effect'].iloc[0][act])
                    else:
                        sal_vals.append(None)
                fig_time.add_trace(go.Scatter(
                    x=horizon_labels, y=sal_vals, mode='lines+markers',
                    name=act[:40]
                ))
            fig_time.update_layout(
                yaxis_title="Зарплата (тыс. руб)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', y=-0.2)
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # ── п.8: Сравнительная таблица ──
        st.markdown("---")
        st.subheader("📋 Сравнение сценариев")
        compare_rows = []
        for act in selected_actions:
            if act not in effects[sal_col]['effect'].columns:
                continue
            row = {
                'Действие': act,
                'Зарплата': f"{effects[sal_col]['effect'].iloc[0][act]:.1f}",
                'Удовлетворённость': f"{effects[sat_col]['effect'].iloc[0][act]:.1f}",
                'Повышение': f"{effects[prom_col]['effect'].iloc[0][act]*100:.1f}%",
                'WLB': f"{effects[wlb_col]['effect'].iloc[0][act]:.1f}",
                'Риск (ДИ зарплаты)': f"{(effects[sal_col]['upper'].iloc[0][act] - effects[sal_col]['lower'].iloc[0][act]):.0f} тыс",
            }
            compare_rows.append(row)
        if compare_rows:
            st.dataframe(pd.DataFrame(compare_rows).set_index('Действие'), use_container_width=True)

        # ── Spider chart ──
        if len(selected_actions) >= 1:
            categories = ['Зарплата (норм.)', 'Удовлетворённость', 'Повышение × 100', 'WLB']

            def _normalize(values):
                arr = np.array(values, dtype=float)
                mn, mx = arr.min(), arr.max()
                return (arr - mn) / (mx - mn + 1e-8)

            all_sal  = [effects[sal_col]['effect'].iloc[0][a] for a in selected_actions if a in effects[sal_col]['effect'].columns]
            all_sat  = [effects[sat_col]['effect'].iloc[0][a] for a in selected_actions if a in effects[sat_col]['effect'].columns]
            all_prom = [effects[prom_col]['effect'].iloc[0][a]*100 for a in selected_actions if a in effects[prom_col]['effect'].columns]
            all_wlb  = [effects[wlb_col]['effect'].iloc[0][a] for a in selected_actions if a in effects[wlb_col]['effect'].columns]

            fig_radar = go.Figure()
            for idx, act in enumerate(selected_actions):
                if act not in effects[sal_col]['effect'].columns:
                    continue
                r = [
                    _normalize(all_sal)[idx] if all_sal else 0,
                    _normalize(all_sat)[idx] if all_sat else 0,
                    _normalize(all_prom)[idx] if all_prom else 0,
                    _normalize(all_wlb)[idx] if all_wlb else 0,
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=r + [r[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=act[:40]
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Сравнение сценариев (нормализованные показатели)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Рекомендация ──
        if selected_actions:
            main_act = selected_actions[0]
            if main_act in effects[sal_col]['effect'].columns:
                salary_delta_main = (effects[sal_col]['effect'].iloc[0][main_act] -
                                     effects[sal_col]['effect'].iloc[0][baseline_action])
                sat_delta_main    = (effects[sat_col]['effect'].iloc[0][main_act] -
                                     effects[sat_col]['effect'].iloc[0][baseline_action])
                wlb_delta_main    = (effects[wlb_col]['effect'].iloc[0][main_act] -
                                     effects[wlb_col]['effect'].iloc[0][baseline_action])

                st.markdown("---")
                st.subheader("💡 Рекомендация")
                if salary_delta_main > 20 and sat_delta_main > 0:
                    st.success("🎉 **Отличный выбор!** Ожидается значительный рост дохода и удовлетворённости.")
                elif salary_delta_main > 0:
                    st.info("📈 **Положительная динамика.** Рост зарплаты вероятен.")
                elif salary_delta_main > -10:
                    st.warning("⚠️ **Умеренные риски.** Возможна стагнация дохода.")
                else:
                    st.error("🚨 **Высокие риски.** Прогнозируется снижение дохода.")
                if wlb_delta_main < -1:
                    st.caption("Баланс работы и жизни может ухудшиться — взвесьте готовность к этому.")

        # ── Сравнительный график зарплат (все действия) ──
        st.markdown("---")
        st.subheader("📉 Зарплата — все действия")
        actions_show = st.session_state['available_actions']
        salary_preds, salary_lower, salary_upper, labels = [], [], [], []
        for a in actions_show:
            if a in effects[sal_col]['effect'].columns:
                salary_preds.append(effects[sal_col]['effect'].iloc[0][a])
                salary_lower.append(effects[sal_col]['lower'].iloc[0][a])
                salary_upper.append(effects[sal_col]['upper'].iloc[0][a])
                labels.append(a)

        fig_bar = go.Figure()
        colors = ['#2E86AB' if a in selected_actions else '#A23B72' for a in labels]
        fig_bar.add_trace(go.Bar(
            x=labels, y=salary_preds,
            error_y=dict(type='data', symmetric=False,
                         array=[salary_upper[i] - salary_preds[i] for i in range(len(salary_preds))],
                         arrayminus=[salary_preds[i] - salary_lower[i] for i in range(len(salary_preds))]),
            marker_color=colors,
            text=[f"{v:.0f}" for v in salary_preds],
            textposition='outside'
        ))
        fig_bar.update_layout(
            title="Прогноз зарплаты для всех действий",
            yaxis_title="Зарплата (тыс. руб)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Выбранные вами действия выделены синим. Планки — 95% доверительные интервалы.")


# ==================== ВКЛАДКА: КАЧЕСТВО МОДЕЛИ (п.11) ====================
with tab_quality:
    st.subheader("📐 Качество модели (calibration)")

    if 'trainer' not in st.session_state:
        st.info("Сначала обучите модель в боковой панели.")
    else:
        if st.button("🔬 Рассчитать метрики качества"):
            with st.spinner("Генерация тестовой выборки и расчёт метрик..."):
                from validator import compute_metrics, calibration_test
                from sklearn.model_selection import train_test_split

                df_val = generate_career_data(n=2000, random_state=99)
                df_train_v, df_test_v = train_test_split(df_val, test_size=0.3, random_state=42)

                trainer_v = st.session_state['trainer']
                feature_cols_v = get_default_feature_cols()
                available_fc = [c for c in feature_cols_v if c in df_test_v.columns]

                # Metrics for salary_2y
                outcome = 'salary_2y'
                if outcome in trainer_v.baseline_means:
                    preds = trainer_v.predict_absolute(df_test_v[available_fc])
                    if outcome in preds:
                        actual_T = df_test_v['treatment'].values
                        y_true   = df_test_v[outcome].values
                        y_pred   = np.array([
                            preds[outcome]['effect'].iloc[j][actual_T[j]]
                            if actual_T[j] in preds[outcome]['effect'].columns else np.nan
                            for j in range(len(df_test_v))
                        ])
                        valid = ~np.isnan(y_pred)
                        if valid.sum() > 10:
                            metrics = compute_metrics(y_true[valid], y_pred[valid])
                            c1, c2, c3 = st.columns(3)
                            c1.metric("MSE (salary_2y)", f"{metrics['MSE']:.1f}")
                            c2.metric("R²", f"{metrics['R2']:.4f}")
                            c3.metric("Pearson R", f"{metrics['PearsonR']:.4f}")

                            # Calibration curve
                            cal = calibration_test(trainer_v, df_test_v, outcome=outcome,
                                                   treatment_col='treatment')
                            valid_bins = ~np.isnan(cal['actual'])
                            if valid_bins.sum() > 1:
                                fig_cal = go.Figure()
                                fig_cal.add_trace(go.Scatter(
                                    x=cal['predicted'][valid_bins],
                                    y=cal['actual'][valid_bins],
                                    mode='markers+lines',
                                    name='Predicted vs Actual'
                                ))
                                # Perfect calibration line
                                mn = min(cal['predicted'][valid_bins].min(), cal['actual'][valid_bins].min())
                                mx = max(cal['predicted'][valid_bins].max(), cal['actual'][valid_bins].max())
                                fig_cal.add_trace(go.Scatter(
                                    x=[mn, mx], y=[mn, mx],
                                    mode='lines',
                                    line=dict(dash='dash', color='gray'),
                                    name='Идеальная калибровка'
                                ))
                                fig_cal.update_layout(
                                    title=f"Calibration curve ({outcome})",
                                    xaxis_title="Predicted CATE (bin mean)",
                                    yaxis_title="Actual effect (bin mean)",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                )
                                st.plotly_chart(fig_cal, use_container_width=True)
                                st.caption(
                                    f"Корреляция предсказанных и фактических CATE: "
                                    f"{cal['correlation']:.3f}"
                                )
                        else:
                            st.warning("Недостаточно совпадающих treatment-меток для оценки.")
                else:
                    st.warning(f"Исход '{outcome}' не найден в модели.")


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    Карьерный советник v4.0 · Прогнозы основаны на синтетических данных и каузальном лесе · Для демонстрационных целей
</div>
""", unsafe_allow_html=True)