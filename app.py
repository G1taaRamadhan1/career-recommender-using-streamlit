"""
Career Path AI - Intelligent Career Recommendation System
==========================================================
Professional Streamlit application for student career guidance.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

# ==================================================================
# PAGE CONFIGURATION
# ==================================================================
st.set_page_config(
    page_title="Career Path AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Career Path AI — Intelligent career recommendation system powered by Machine Learning."
    }
)


# ==================================================================
# CUSTOM CSS - Professional Theme
# ==================================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Keep header transparent so sidebar toggle button stays accessible */
    header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
    display: none;
    }

    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Hero section styling */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .hero-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Section headings */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #1a202c;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        display: inline-block;
    }

    /* Custom metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #667eea;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 500;
        color: #64748b;
    }

    /* Info boxes - more professional */
    .info-card {
        background: linear-gradient(135deg, #f6f9fc 0%, #e9efff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .info-card h3 {
        color: #1a202c;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .info-card p {
        color: #475569;
        line-height: 1.6;
        margin: 0;
    }

    /* Recommendation cards */
    .recommendation-card {
        background: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 0.6rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    .recommendation-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    .recommendation-card-match {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
    }
    .rec-rank {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        text-align: center;
        background: #667eea;
        color: white;
        border-radius: 50%;
        font-weight: 700;
        margin-right: 1rem;
    }
    .rec-rank-match {
        background: #10b981;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    [data-testid="stSidebar"] .sidebar-content {
        padding-top: 2rem;
    }

    /* Radio buttons - cleaner look */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: white;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #f1f5f9;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem 1rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    .custom-footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #667eea !important;
        color: white !important;
    }

    /* Form */
    .stForm {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

    /* Progress bars in app */
    .progress-bar-container {
        background: #e2e8f0;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin: 0.3rem 0;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================================================================
# DATA LOADING & PREPARATION
# ==================================================================
@st.cache_data
def load_and_prepare_data():
    """Load dataset and prepare for modeling."""
    df = pd.read_csv("student-scores.csv")
    df = df[df["career_aspiration"] != "Unknown"].reset_index(drop=True)

    df_clean = df.drop(columns=["first_name", "last_name", "email"]).copy()
    df_clean["gender"] = df_clean["gender"].map({"male": 0, "female": 1})
    df_clean["part_time_job"] = df_clean["part_time_job"].astype(int)
    df_clean["extracurricular_activities"] = df_clean["extracurricular_activities"].astype(int)

    feature_cols = [
        "gender", "part_time_job", "absence_days", "extracurricular_activities",
        "weekly_self_study_hours", "math_score", "history_score", "physics_score",
        "chemistry_score", "biology_score", "english_score", "geography_score"
    ]

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df_clean[feature_cols])

    career_encoder = LabelEncoder()
    df_clean["career_id"] = career_encoder.fit_transform(df_clean["career_aspiration"])

    return df, df_clean, features_scaled, feature_cols, scaler, career_encoder


@st.cache_resource
def compute_similarity(features_scaled):
    """Compute cosine similarity matrix (cached)."""
    return cosine_similarity(features_scaled)


# ==================================================================
# RECOMMENDATION FUNCTIONS
# ==================================================================
def recommend_career_for_existing(student_pos, df_clean, sim_matrix,
                                   top_n=5, top_k_neighbors=50):
    """Recommend careers for existing student."""
    sims = sim_matrix[student_pos].copy()
    sims[student_pos] = -np.inf
    neighbor_idx = sims.argsort()[::-1][:top_k_neighbors]
    neighbor_careers = df_clean.iloc[neighbor_idx]["career_aspiration"].values
    neighbor_sims = sim_matrix[student_pos][neighbor_idx]

    score_dict = {}
    for c, s in zip(neighbor_careers, neighbor_sims):
        score_dict[c] = score_dict.get(c, 0.0) + s

    sorted_careers = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    total = sum([s for _, s in sorted_careers]) or 1.0
    return pd.DataFrame([
        {"Career": c, "Score": round(s, 4), "Confidence (%)": round(100 * s / total, 2)}
        for c, s in sorted_careers[:top_n]
    ])


def recommend_career_for_new(profile, df_clean, features_scaled, scaler,
                              top_n=5, top_k_neighbors=50):
    """Recommend careers for new student based on input profile."""
    profile_scaled = scaler.transform([profile])[0]
    sims = cosine_similarity([profile_scaled], features_scaled)[0]

    neighbor_idx = sims.argsort()[::-1][:top_k_neighbors]
    neighbor_careers = df_clean.iloc[neighbor_idx]["career_aspiration"].values
    neighbor_sims = sims[neighbor_idx]

    score_dict = {}
    for c, s in zip(neighbor_careers, neighbor_sims):
        score_dict[c] = score_dict.get(c, 0.0) + s

    sorted_careers = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    total = sum([s for _, s in sorted_careers]) or 1.0
    return pd.DataFrame([
        {"Career": c, "Score": round(s, 4), "Confidence (%)": round(100 * s / total, 2)}
        for c, s in sorted_careers[:top_n]
    ])


# ==================================================================
# UI COMPONENTS
# ==================================================================
def render_hero(title, subtitle):
    """Render hero section with gradient background."""
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation_card(rank, career, confidence, is_match=False, is_top=False):
    """Render single recommendation card."""
    card_class = "recommendation-card-match" if is_match else "recommendation-card"
    rank_class = "rec-rank-match" if is_match else "rec-rank"
    badge = " 🎯 Match!" if is_match else ""
    star = " ⭐" if is_top and not is_match else ""

    st.markdown(f"""
    <div class="recommendation-card {card_class}">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; flex: 1;">
                <span class="rec-rank {rank_class}">{rank}</span>
                <div>
                    <div style="font-weight: 600; font-size: 1.1rem; color: #1a202c;">
                        {career}{badge}{star}
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {min(confidence, 100)}%;"></div>
                    </div>
                </div>
            </div>
            <div style="font-weight: 700; font-size: 1.2rem; color: #667eea; margin-left: 1rem;">
                {confidence}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================================
# MAIN PAGES
# ==================================================================
def page_home(df, df_clean):
    """Home page with overview and stats."""
    render_hero(
        "🎯 Career Path AI",
        "Discover your perfect career path through intelligent profile-based recommendations powered by machine learning."
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Students Analyzed", f"{len(df_clean):,}")
    with col2:
        st.metric("🎓 Career Paths", df_clean["career_aspiration"].nunique())
    with col3:
        st.metric("📊 Subjects Tracked", "7")
    with col4:
        st.metric("🎯 Accuracy", "94.6%", help="Hit Rate at Top-10 recommendations")

    st.markdown('<h2 class="section-title">How It Works</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>1️⃣ Input Profile</h3>
            <p>Enter academic scores, study habits, and personal attributes that define your unique profile.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>2️⃣ AI Analysis</h3>
            <p>Our model analyzes your profile against thousands of similar students using cosine similarity algorithm.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>3️⃣ Get Recommendations</h3>
            <p>Receive personalized career recommendations ranked by confidence, helping you make informed decisions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Top Career Aspirations</h2>', unsafe_allow_html=True)
    career_counts = df_clean["career_aspiration"].value_counts().head(10)
    st.bar_chart(career_counts, color="#667eea")

    st.markdown('<h2 class="section-title">Get Started</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🔍 Browse Existing Profiles</h3>
            <p>Explore career recommendations for students already in our database. Perfect for understanding how the system works.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>✨ Get Your Recommendations</h3>
            <p>Input your own academic profile and discover career paths tailored to your unique strengths and interests.</p>
        </div>
        """, unsafe_allow_html=True)


def page_search_student(df, df_clean, sim_matrix):
    """Search and recommend for existing students."""
    render_hero(
        "🔍 Student Profile Explorer",
        "Browse student profiles from our database and view personalized career recommendations."
    )

    # Search method
    col1, col2 = st.columns([2, 1])
    with col1:
        search_method = st.radio(
            "Search by:",
            ["Student ID", "Full Name"],
            horizontal=True
        )
    with col2:
        top_n = st.slider("Number of recommendations", 3, 10, 5)

    # Search input
    if search_method == "Student ID":
        student_id = st.number_input(
            "Enter Student ID",
            min_value=int(df["id"].min()),
            max_value=int(df["id"].max()),
            value=1, step=1
        )
        student_row = df[df["id"] == student_id]
    else:
        df["full_name"] = df["first_name"] + " " + df["last_name"]
        selected_name = st.selectbox(
            "Select student",
            options=df["full_name"].sort_values().tolist()
        )
        student_row = df[df["full_name"] == selected_name]
        student_id = student_row["id"].values[0]

    if student_row.empty:
        st.warning("⚠️ Student not found.")
        return

    sr = student_row.iloc[0]

    # Profile section
    st.markdown('<h2 class="section-title">📋 Student Profile</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h3>👤 Personal Info</h3>
            <p>
                <b>Name:</b> {sr['first_name']} {sr['last_name']}<br>
                <b>ID:</b> {sr['id']}<br>
                <b>Gender:</b> {sr['gender'].title()}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h3>📊 Academic Behavior</h3>
            <p>
                <b>Study hours/week:</b> {sr['weekly_self_study_hours']} hours<br>
                <b>Absence days:</b> {sr['absence_days']} days<br>
                <b>Extracurricular:</b> {'Yes' if sr['extracurricular_activities'] else 'No'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <h3>🎯 Aspiration</h3>
            <p>
                <b>Part-time job:</b> {'Yes' if sr['part_time_job'] else 'No'}<br>
                <b>Career goal:</b><br>
                <span style="font-size: 1.2rem; font-weight: 700; color: #667eea;">{sr['career_aspiration']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Scores chart
    st.markdown('<h3 style="margin-top: 1.5rem;">📚 Academic Scores</h3>', unsafe_allow_html=True)
    score_data = {
        "Mathematics": sr["math_score"],
        "History": sr["history_score"],
        "Physics": sr["physics_score"],
        "Chemistry": sr["chemistry_score"],
        "Biology": sr["biology_score"],
        "English": sr["english_score"],
        "Geography": sr["geography_score"]
    }
    score_df = pd.DataFrame.from_dict(score_data, orient="index", columns=["Score"])
    st.bar_chart(score_df, color="#667eea")

    # Recommendations
    st.markdown('<h2 class="section-title">✨ Career Recommendations</h2>', unsafe_allow_html=True)

    if student_id in df_clean["id"].values:
        student_pos = df_clean.index[df_clean["id"] == student_id][0]
        rec = recommend_career_for_existing(student_pos, df_clean, sim_matrix, top_n=top_n)

        for i, row in rec.iterrows():
            is_match = row["Career"] == sr["career_aspiration"]
            render_recommendation_card(
                rank=i+1,
                career=row["Career"],
                confidence=row["Confidence (%)"],
                is_match=is_match,
                is_top=(i == 0)
            )

        with st.expander("📊 View detailed scores"):
            st.dataframe(rec, use_container_width=True, hide_index=True)
    else:
        st.warning("This student has 'Unknown' career aspiration and cannot be recommended.")


def page_input_profile(df_clean, features_scaled, scaler):
    """Input new student profile and get recommendations."""
    render_hero(
        "✨ Get Your Career Recommendations",
        "Input your academic profile and study habits to discover career paths tailored just for you."
    )

    # Initialize session state
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "current_profile" not in st.session_state:
        st.session_state.current_profile = None

    with st.form("new_student_form"):
        st.markdown("### 📝 Tell Us About Yourself")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 👤 Personal Profile")
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            part_time = st.radio("Do you have a part-time job?", ["No", "Yes"], horizontal=True)
            ekskul = st.radio("Are you involved in extracurricular activities?", ["No", "Yes"], horizontal=True)
            absence = st.slider("Absence days this semester", 0, 10, 3)
            study_hours = st.slider("Self-study hours per week", 0, 50, 15)

        with col2:
            st.markdown("#### 📚 Academic Performance (0-100)")
            math = st.slider("Mathematics", 50, 100, 75)
            history = st.slider("History", 50, 100, 75)
            physics = st.slider("Physics", 50, 100, 75)
            chemistry = st.slider("Chemistry", 50, 100, 75)
            biology = st.slider("Biology", 50, 100, 75)
            english = st.slider("English", 50, 100, 75)
            geography = st.slider("Geography", 50, 100, 75)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            top_n_new = st.slider("Number of career recommendations", 3, 10, 5)
        with col_b:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("🎯 Analyze My Profile", type="primary",
                                              use_container_width=True)

    if submitted:
        profile = [
            1 if gender == "Female" else 0,
            1 if part_time == "Yes" else 0,
            absence,
            1 if ekskul == "Yes" else 0,
            study_hours,
            math, history, physics, chemistry, biology, english, geography
        ]
        st.session_state.current_profile = profile
        st.session_state.current_top_n = top_n_new
        st.session_state.show_results = True

    if st.session_state.show_results and st.session_state.current_profile:
        st.markdown('<h2 class="section-title">🎉 Your Career Recommendations</h2>', unsafe_allow_html=True)

        rec = recommend_career_for_new(
            st.session_state.current_profile,
            df_clean, features_scaled, scaler,
            top_n=st.session_state.current_top_n
        )

        # Top recommendation highlight
        top_career = rec.iloc[0]["Career"]
        top_confidence = rec.iloc[0]["Confidence (%)"]

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem; border-radius: 16px; color: white; text-align: center;
                    margin: 1.5rem 0; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);">
            <div style="font-size: 1rem; opacity: 0.9; margin-bottom: 0.5rem;">🏆 TOP RECOMMENDATION</div>
            <div style="font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;">{top_career}</div>
            <div style="font-size: 1.1rem; opacity: 0.95;">Confidence: <b>{top_confidence}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Complete Top-{} Recommendations".format(st.session_state.current_top_n))

        for i, row in rec.iterrows():
            render_recommendation_card(
                rank=i+1,
                career=row["Career"],
                confidence=row["Confidence (%)"],
                is_top=(i == 0)
            )

        with st.expander("📊 View detailed scores"):
            st.dataframe(rec, use_container_width=True, hide_index=True)

        st.info("💡 **Disclaimer**: These recommendations are AI-generated suggestions based on profile similarity. Use them as a starting point for career exploration alongside professional career counseling.")


def page_explore(df_clean):
    """Data exploration page."""
    render_hero(
        "📊 Dataset Insights",
        "Explore the underlying data, patterns, and statistics that power our recommendation engine."
    )

    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🎯 Career Distribution", "📚 Subject Analysis"])

    with tab1:
        st.markdown("### Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", len(df_clean))
        col2.metric("Career Paths", df_clean["career_aspiration"].nunique())
        col3.metric("Avg Math Score", f"{df_clean['math_score'].mean():.1f}")
        col4.metric("Avg Study Hours/Week", f"{df_clean['weekly_self_study_hours'].mean():.1f}")

        score_cols = ["math_score", "history_score", "physics_score", "chemistry_score",
                      "biology_score", "english_score", "geography_score"]
        st.markdown("### Score Statistics")
        st.dataframe(df_clean[score_cols].describe().round(2), use_container_width=True)

    with tab2:
        st.markdown("### Career Distribution")
        career_counts = df_clean["career_aspiration"].value_counts()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(career_counts, color="#667eea")
        with col2:
            st.dataframe(
                career_counts.reset_index().rename(
                    columns={"career_aspiration": "Career", "count": "Count"}
                ),
                use_container_width=True, hide_index=True
            )

    with tab3:
        st.markdown("### Subject Score Distribution")
        score_cols = ["math_score", "history_score", "physics_score", "chemistry_score",
                      "biology_score", "english_score", "geography_score"]
        score_labels = {
            "math_score": "Mathematics",
            "history_score": "History",
            "physics_score": "Physics",
            "chemistry_score": "Chemistry",
            "biology_score": "Biology",
            "english_score": "English",
            "geography_score": "Geography"
        }
        chosen = st.selectbox("Select subject", score_cols, format_func=lambda x: score_labels[x])
        st.bar_chart(df_clean[chosen].value_counts().sort_index(), color="#667eea")

        st.markdown("### Subject Correlation Matrix")
        st.dataframe(df_clean[score_cols].corr().round(3), use_container_width=True)


# ==================================================================
# MAIN APP
# ==================================================================
def main():
    inject_custom_css()

    df, df_clean, features_scaled, feature_cols, scaler, career_encoder = load_and_prepare_data()
    sim_matrix = compute_similarity(features_scaled)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size: 3rem;">🎯</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #1a202c;
                        margin-top: 0.5rem;">Career Path AI</div>
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">
                Intelligent Career Guidance
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Choose a page",
            ["🏠 Home", "🔍 Browse Profiles", "✨ Get Recommendations", "📊 Data Insights"],
            label_visibility="collapsed"
        )

        st.divider()

        st.markdown("### ℹ️ About")
        st.info(
            "Career Path AI uses **Content-Based Filtering** with cosine similarity "
            "to recommend careers based on student profiles."
        )

        st.markdown("### 📈 Performance")
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.3rem;">
                Top-10 Accuracy
            </div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #10b981;">94.6%</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
            Made by Gita Ramadhani W.S using<br>
            <b>Python</b> · <b>scikit-learn</b> · <b>Streamlit</b><br><br>
            © 2026 Career Path AI<br>
            Licensed under MIT
        </div>
        """, unsafe_allow_html=True)

    # Routing
    if page == "🏠 Home":
        page_home(df, df_clean)
    elif page == "🔍 Browse Profiles":
        page_search_student(df, df_clean, sim_matrix)
    elif page == "✨ Get Recommendations":
        page_input_profile(df_clean, features_scaled, scaler)
    elif page == "📊 Data Insights":
        page_explore(df_clean)

    # Footer
    st.markdown("""
    <div class="custom-footer">
        <p>
            <b>Career Path AI</b> — Empowering students to discover their potential.<br>
            Built with modern ML technology · 
            <a href="https://github.com" target="_blank">View on GitHub</a> · 
            Licensed under MIT
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
