import os
import math
import random

import pandas as pd
import streamlit as st
from crewai import Agent, Crew, LLM, Task


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Smart Delivery AI",
    page_icon="🚚",
    layout="wide",
)


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #ffffff, #eef6ff);
        border: 1px solid #e5eaf0;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 750;
        color: #172033;
    }

    .hero-subtitle {
        color: #667085;
        font-size: 16px;
        margin-top: 8px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e6eaf0;
        border-radius: 16px;
        padding: 18px;
        min-height: 105px;
        box-shadow: 0 5px 18px rgba(20,40,80,.04);
    }

    .metric-label {
        color: #667085;
        font-size: 13px;
    }

    .metric-value {
        color: #172033;
        font-size: 27px;
        font-weight: 750;
        margin-top: 5px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #172033;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .route-box {
        background: white;
        border: 1px solid #e6eaf0;
        border-radius: 16px;
        padding: 20px;
    }

    .route-item {
        display: inline-block;
        background: #eef6ff;
        color: #175cd3;
        padding: 9px 13px;
        margin: 4px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 650;
    }

    .agent-box {
        background: white;
        border: 1px solid #e6eaf0;
        border-radius: 16px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SETTINGS
# =========================================================

DEPOT = (50, 50)

# Groq model through LiteLLM/CrewAI
GROQ_MODEL = "groq/llama-3.1-8b-instant"


# =========================================================
# API KEY
# =========================================================

def get_groq_key():

    try:
        key = st.secrets.get("GROQ_API_KEY")

        if key:
            return key

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


# =========================================================
# LLM
# =========================================================

def get_llm():

    api_key = get_groq_key()

    if not api_key:
        return None

    return LLM(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.2,
        max_tokens=800,
    )


# =========================================================
# PACKAGE GENERATOR
# =========================================================

def generate_packages(number_of_packages, seed):

    rng = random.Random(seed)

    packages = []

    for i in range(number_of_packages):

        priority = rng.choice(
            [
                "Normal",
                "Normal",
                "High",
                "Urgent",
            ]
        )

        if priority == "Urgent":
            priority_score = 3
            deadline = rng.randint(45, 70)

        elif priority == "High":
            priority_score = 2
            deadline = rng.randint(60, 100)

        else:
            priority_score = 1
            deadline = rng.randint(80, 130)

        packages.append(
            {
                "id": i,
                "name": f"Package {chr(65 + i)}",
                "x": rng.randint(5, 95),
                "y": rng.randint(5, 95),
                "priority": priority,
                "priority_score": priority_score,
                "deadline": deadline,
                "weight": round(
                    rng.uniform(0.5, 5.0),
                    1,
                ),
            }
        )

    return packages


# =========================================================
# DISTANCE
# =========================================================

def distance(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2
        +
        (a[1] - b[1]) ** 2
    )


# =========================================================
# REWARD
# =========================================================

def calculate_reward(
    package,
    current_location,
    current_time,
    traffic_multiplier,
    speed_weight,
    cost_weight,
    priority_weight,
):

    target = (
        package["x"],
        package["y"],
    )

    dist = distance(
        current_location,
        target,
    )

    travel_time = (
        dist * traffic_multiplier
    )

    arrival_time = (
        current_time + travel_time
    )

    late_minutes = max(
        0,
        arrival_time - package["deadline"],
    )

    priority_reward = (
        package["priority_score"]
        * 20
        * priority_weight
    )

    distance_cost = (
        dist * cost_weight
    )

    time_cost = (
        travel_time * speed_weight
    )

    late_penalty = (
        late_minutes * 5
    )

    if (
        package["priority"] == "Urgent"
        and late_minutes > 0
    ):
        late_penalty += 50

    reward = (
        100
        + priority_reward
        - distance_cost
        - time_cost
        - late_penalty
    )

    return reward, {
        "distance": dist,
        "travel_time": travel_time,
        "arrival_time": arrival_time,
        "late_minutes": late_minutes,
    }


# =========================================================
# Q LEARNING
# =========================================================

def train_q_learning(
    packages,
    episodes,
    speed_weight,
    cost_weight,
    priority_weight,
    traffic_multiplier,
):

    q_table = {}
    history = []

    lookup = {
        p["id"]: p
        for p in packages
    }

    learning_rate = 0.15
    discount = 0.90

    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05

    for _ in range(episodes):

        remaining = tuple(
            p["id"]
            for p in packages
        )

        current_location = DEPOT
        current_package = -1
        current_time = 0
        total_reward = 0

        while remaining:

            state = (
                current_package,
                remaining,
            )

            if state not in q_table:
                q_table[state] = {}

            for action in remaining:

                if action not in q_table[state]:
                    q_table[state][action] = 0.0

            actions = list(remaining)

            if random.random() < epsilon:

                action = random.choice(actions)

            else:

                action = max(
                    actions,
                    key=lambda x:
                    q_table[state][x],
                )

            package = lookup[action]

            reward, details = calculate_reward(
                package,
                current_location,
                current_time,
                traffic_multiplier,
                speed_weight,
                cost_weight,
                priority_weight,
            )

            total_reward += reward

            next_remaining = tuple(
                x
                for x in remaining
                if x != action
            )

            next_state = (
                action,
                next_remaining,
            )

            if next_remaining:

                if next_state not in q_table:
                    q_table[next_state] = {}

                for next_action in next_remaining:

                    if (
                        next_action
                        not in q_table[next_state]
                    ):
                        q_table[next_state][next_action] = 0.0

                best_next = max(
                    q_table[next_state][x]
                    for x in next_remaining
                )

            else:

                best_next = 0

            old_value = q_table[state][action]

            q_table[state][action] = (
                old_value
                +
                learning_rate
                *
                (
                    reward
                    +
                    discount * best_next
                    -
                    old_value
                )
            )

            current_location = (
                package["x"],
                package["y"],
            )

            current_time = details["arrival_time"]
            current_package = action
            remaining = next_remaining

        history.append(total_reward)

        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay,
        )

    return q_table, history


# =========================================================
# GET LEARNED ROUTE
# =========================================================

def get_learned_route(
    q_table,
    packages,
    speed_weight,
    cost_weight,
    priority_weight,
    traffic_multiplier,
):

    lookup = {
        p["id"]: p
        for p in packages
    }

    remaining = tuple(
        p["id"]
        for p in packages
    )

    current_location = DEPOT
    current_package = -1
    current_time = 0

    route = []
    rows = []

    total_reward = 0
    total_distance = 0
    total_time = 0
    late_deliveries = 0

    while remaining:

        state = (
            current_package,
            remaining,
        )

        actions = list(remaining)

        if state in q_table:

            action = max(
                actions,
                key=lambda x:
                q_table[state].get(
                    x,
                    0,
                ),
            )

        else:

            action = min(
                actions,
                key=lambda x:
                distance(
                    current_location,
                    (
                        lookup[x]["x"],
                        lookup[x]["y"],
                    ),
                ),
            )

        package = lookup[action]

        reward, details = calculate_reward(
            package,
            current_location,
            current_time,
            traffic_multiplier,
            speed_weight,
            cost_weight,
            priority_weight,
        )

        route.append(package)

        total_reward += reward
        total_distance += details["distance"]
        total_time += details["travel_time"]

        if details["late_minutes"] > 0:
            late_deliveries += 1

        rows.append(
            {
                "Package": package["name"],
                "Priority": package["priority"],
                "Distance": round(
                    details["distance"],
                    2,
                ),
                "Travel Time": round(
                    details["travel_time"],
                    2,
                ),
                "Arrival": round(
                    details["arrival_time"],
                    2,
                ),
                "Deadline": package["deadline"],
                "Late": (
                    details["late_minutes"] > 0
                ),
                "Reward": round(
                    reward,
                    2,
                ),
            }
        )

        current_location = (
            package["x"],
            package["y"],
        )

        current_time = details["arrival_time"]
        current_package = action

        remaining = tuple(
            x
            for x in remaining
            if x != action
        )

    return {
        "route": route,
        "rows": rows,
        "total_reward": total_reward,
        "total_distance": total_distance,
        "total_time": total_time,
        "late_deliveries": late_deliveries,
    }


# =========================================================
# CREWAI PLANNER
# =========================================================

def run_planner(packages):

    llm = get_llm()

    if llm is None:

        return (
            "Groq API key is not configured. "
            "RL can still run, but CrewAI requires "
            "the Groq API key."
        )

    package_text = "\n".join(
        [
            (
                f"{p['name']} | "
                f"Priority={p['priority']} | "
                f"Deadline={p['deadline']} min | "
                f"Location=({p['x']},{p['y']})"
            )
            for p in packages
        ]
    )

    agent = Agent(
        role="Delivery Planning Agent",
        goal=(
            "Analyze package urgency, deadlines "
            "and delivery risks."
        ),
        backstory=(
            "You are a logistics planner helping "
            "an RL delivery system understand "
            "the delivery situation."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
        Analyze the following packages:

        {package_text}

        Identify:
        - urgent packages
        - important deadlines
        - possible delivery risks
        - important factors for the RL agent

        Do not calculate the final route.

        Keep the answer concise.
        """,
        expected_output=(
            "A concise logistics analysis."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    try:
        result = crew.kickoff()
        return str(result)

    except Exception as e:

        return (
            "CrewAI error: "
            + str(e)
        )


# =========================================================
# CREWAI EXPLANATION
# =========================================================

def explain_route(
    result,
    planner_analysis,
):

    llm = get_llm()

    if llm is None:
        return "Groq API key is not configured."

    route = " → ".join(
        p["name"]
        for p in result["route"]
    )

    agent = Agent(
        role="Delivery Decision Analyst",
        goal=(
            "Explain the delivery route "
            "in simple language."
        ),
        backstory=(
            "You explain AI delivery decisions "
            "to logistics operators."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
        Explain this delivery decision.

        Planner analysis:
        {planner_analysis}

        Learned route:
        {route}

        Reward:
        {result["total_reward"]:.2f}

        Distance:
        {result["total_distance"]:.2f}

        Time:
        {result["total_time"]:.2f}

        Late deliveries:
        {result["late_deliveries"]}

        Explain how priority, time, distance,
        cost and deadlines affected the result.

        Do not claim the route is globally optimal.

        Keep the answer simple.
        """,
        expected_output=(
            "A short explanation of the route."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
    )

    try:
        result = crew.kickoff()
        return str(result)

    except Exception as e:

        return (
            "CrewAI error: "
            + str(e)
        )


# =========================================================
# SESSION
# =========================================================

if "packages" not in st.session_state:

    st.session_state.packages = (
        generate_packages(6, 42)
    )

if "result" not in st.session_state:

    st.session_state.result = None

if "history" not in st.session_state:

    st.session_state.history = []

if "planner" not in st.session_state:

    st.session_state.planner = ""


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🚚 Smart Delivery AI Agent
        </div>

        <div class="hero-subtitle">
            CrewAI + Groq + Q-Learning
            <br>
            Intelligent package delivery
            using reward, time, cost and priority.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Delivery Settings")

    number_of_packages = st.slider(
        "Packages",
        4,
        8,
        6,
    )

    episodes = st.slider(
        "RL Training Episodes",
        500,
        3000,
        1500,
        500,
    )

    speed_weight = st.slider(
        "Speed",
        0.1,
        3.0,
        1.0,
        0.1,
    )

    cost_weight = st.slider(
        "Cost",
        0.1,
        3.0,
        1.0,
        0.1,
    )

    priority_weight = st.slider(
        "Priority",
        0.1,
        3.0,
        1.5,
        0.1,
    )

    traffic = st.selectbox(
        "Traffic",
        [
            "Normal",
            "Moderate",
            "Heavy",
        ],
    )

    traffic_multiplier = {
        "Normal": 1.0,
        "Moderate": 1.25,
        "Heavy": 1.60,
    }[traffic]

    st.divider()

    new_scenario = st.button(
        "🔄 New Scenario",
        use_container_width=True,
    )

    run_agent = st.button(
        "🧠 Run AI Agent",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# NEW SCENARIO
# =========================================================

if new_scenario:

    st.session_state.packages = (
        generate_packages(
            number_of_packages,
            random.randint(
                1,
                999999,
            ),
        )
    )

    st.session_state.result = None
    st.session_state.history = []
    st.session_state.planner = ""

    st.rerun()


# =========================================================
# RUN
# =========================================================

if run_agent:

    packages = st.session_state.packages

    progress = st.progress(
        0,
        text="Starting AI...",
    )

    progress.progress(
        10,
        text="CrewAI is analyzing packages...",
    )

    planner = run_planner(packages)

    st.session_state.planner = planner

    progress.progress(
        30,
        text="Training Q-Learning agent...",
    )

    q_table, history = train_q_learning(
        packages,
        episodes,
        speed_weight,
        cost_weight,
        priority_weight,
        traffic_multiplier,
    )

    progress.progress(
        85,
        text="Generating learned route...",
    )

    result = get_learned_route(
        q_table,
        packages,
        speed_weight,
        cost_weight,
        priority_weight,
        traffic_multiplier,
    )

    st.session_state.result = result
    st.session_state.history = history

    progress.progress(
        100,
        text="Completed.",
    )

    st.success(
        "CrewAI + RL delivery planning completed."
    )


# =========================================================
# PACKAGES
# =========================================================

packages = st.session_state.packages
result = st.session_state.result

st.markdown(
    '<div class="section-title">📦 Packages</div>',
    unsafe_allow_html=True,
)

package_df = pd.DataFrame(
    [
        {
            "Package": p["name"],
            "Priority": p["priority"],
            "Deadline": f'{p["deadline"]} min',
            "Weight": f'{p["weight"]} kg',
            "Location": f'({p["x"]}, {p["y"]})',
        }
        for p in packages
    ]
)

st.dataframe(
    package_df,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# RESULTS
# =========================================================

if result is not None:

    total = len(packages)

    on_time = (
        total - result["late_deliveries"]
    )

    on_time_percent = (
        on_time / total * 100
    )

    estimated_cost = (
        result["total_distance"]
        * cost_weight
    )

    st.markdown(
        '<div class="section-title">'
        '📊 AI Results'
        '</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)

    metrics = [
        ("📦 Packages", total, "Total"),
        (
            "🏆 Reward",
            f'{result["total_reward"]:.0f}',
            "RL reward",
        ),
        (
            "📍 Distance",
            f'{result["total_distance"]:.1f}',
            "Units",
        ),
        (
            "💰 Cost",
            f"{estimated_cost:.1f}",
            "Estimated",
        ),
        (
            "⏱️ On Time",
            f"{on_time_percent:.0f}%",
            "Performance",
        ),
    ]

    for col, metric in zip(cols, metrics):

        label, value, description = metric

        col.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    {label}
                </div>

                <div class="metric-value">
                    {value}
                </div>

                <div style="color:#98a2b3;font-size:12px;">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    # =====================================================
    # ROUTE
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🛣️ Learned Route'
        '</div>',
        unsafe_allow_html=True,
    )

    route_html = (
        '<div class="route-box">'
        '<span class="route-item">'
        '🏠 Depot'
        '</span>'
    )

    for p in result["route"]:

        route_html += (
            '<span> → </span>'
            '<span class="route-item">'
            f'📦 {p["name"]}'
            '</span>'
        )

    route_html += (
        '<span> → </span>'
        '<span class="route-item">'
        '🏁 Complete'
        '</span>'
        '</div>'
    )

    st.markdown(
        route_html,
        unsafe_allow_html=True,
    )


    # =====================================================
    # DETAILS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Delivery Details'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        pd.DataFrame(result["rows"]),
        use_container_width=True,
        hide_index=True,
    )


    # =====================================================
    # LEARNING
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🧠 RL Learning Progress'
        '</div>',
        unsafe_allow_html=True,
    )

    history = pd.Series(
        st.session_state.history
    )

    window = min(
        100,
        len(history),
    )

    average = history.rolling(
        window,
        min_periods=1,
    ).mean()

    chart = pd.DataFrame(
        {
            "Episode Reward": history,
            "Average Reward": average,
        }
    )

    st.line_chart(
        chart,
        height=300,
    )


    # =====================================================
    # CREWAI
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🤖 CrewAI Planner'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="agent-box">
        {st.session_state.planner}
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="section-title">'
        '💬 AI Route Explanation'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("🤖 Explain My Route"):

        with st.spinner(
            "CrewAI is explaining the route..."
        ):

            explanation = explain_route(
                result,
                st.session_state.planner,
            )

        st.markdown(
            f"""
            <div class="agent-box">
            {explanation}
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <br><br>
    <center>
    Smart Delivery AI Agent
    <br>
    CrewAI + Groq + Q-Learning + Streamlit
    </center>
    """,
    unsafe_allow_html=True,
)
