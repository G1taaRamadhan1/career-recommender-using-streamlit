"""
Streamlit App: Sistem Rekomendasi Karier untuk Siswa
=====================================================
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
# KONFIGURASI HALAMAN
# ==================================================================
st.set_page_config(
    page_title="Sistem Rekomendasi Karier Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================================
# DATA LOADING & PREPARATION (di-cache supaya tidak load berulang)
# ==================================================================
@st.cache_data
def load_and_prepare_data():
    """Load dataset dan siapkan untuk modeling."""
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
    """Hitung cosine similarity matrix (cached)."""
    return cosine_similarity(features_scaled)


# ==================================================================
# REKOMENDASI FUNCTIONS
# ==================================================================
def recommend_career_cbf_for_existing(student_pos, df_clean, sim_matrix,
                                      top_n=5, top_k_neighbors=50):
    """Rekomendasi karier untuk siswa yang sudah ada di dataset."""
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
        {"Karier": c, "Skor": round(s, 4), "Kepercayaan (%)": round(100 * s / total, 2)}
        for c, s in sorted_careers[:top_n]
    ])


def recommend_career_for_new_student(profile, df_clean, features_scaled, scaler,
                                      feature_cols, top_n=5, top_k_neighbors=50):
    """Rekomendasi karier untuk siswa baru (input manual)."""
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
        {"Karier": c, "Skor": round(s, 4), "Kepercayaan (%)": round(100 * s / total, 2)}
        for c, s in sorted_careers[:top_n]
    ])


# ==================================================================
# UI UTAMA
# ==================================================================
def main():
    # Load data
    df, df_clean, features_scaled, feature_cols, scaler, career_encoder = load_and_prepare_data()
    sim_matrix = compute_similarity(features_scaled)

    # ====== HEADER ======
    st.title("🎓 Sistem Rekomendasi Karier untuk Siswa")
    st.markdown("""
    Aplikasi ini membantu siswa SMA menemukan karier yang cocok berdasarkan profil akademik mereka.
    Sistem ini menggunakan **Content-Based Filtering** dengan cosine similarity.
    """)
    st.divider()

    # ====== SIDEBAR ======
    st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/3976/3976625.png",
        width=100
    )
    st.sidebar.title("⚙️ Menu")

    mode = st.sidebar.radio(
        "Pilih Mode",
        ["🏠 Beranda", "🔍 Cari Siswa di Dataset", "✏️ Input Profil Siswa Baru", "📊 Eksplorasi Dataset"]
    )

    st.sidebar.divider()
    st.sidebar.markdown("### ℹ️ Tentang")
    st.sidebar.info(
        "Proyek Akhir Machine Learning Terapan - Dicoding\n\n"
        f"**Total siswa**: {len(df_clean)}\n\n"
        f"**Total karier**: {df_clean['career_aspiration'].nunique()}"
    )

    # ====== MODE: BERANDA ======
    if mode == "🏠 Beranda":
        st.subheader("Selamat Datang! 👋")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Siswa", f"{len(df_clean):,}")
        col2.metric("Total Karier", df_clean["career_aspiration"].nunique())
        col3.metric("Mata Pelajaran", "7")

        st.markdown("""
        ### 📋 Cara Pakai

        1. **🔍 Cari Siswa di Dataset** — pilih siswa yang sudah ada di database, lihat rekomendasi karier untuknya
        2. **✏️ Input Profil Siswa Baru** — masukkan nilai-nilai dan profil siswa baru, dapatkan rekomendasi instant
        3. **📊 Eksplorasi Dataset** — lihat statistik dan distribusi data

        ### 🤖 Cara Kerja Sistem

        Sistem ini pakai algoritma **Content-Based Filtering**:
        1. Hitung kemiripan profil antar siswa pakai **cosine similarity**
        2. Cari **50 siswa paling mirip** dengan target
        3. Lakukan **weighted voting** — karier yang paling sering muncul (dengan bobot similarity) jadi rekomendasi

        ### 📈 Performa Model

        - **Hit Rate Top-5**: 76.76%
        - **Hit Rate Top-7**: 86.44%
        - **Hit Rate Top-10**: 94.60%
        """)

        st.divider()

        st.subheader("🎯 Distribusi Karier Populer")
        career_counts = df_clean["career_aspiration"].value_counts().head(10)
        st.bar_chart(career_counts)

    # ====== MODE: CARI SISWA ======
    elif mode == "🔍 Cari Siswa di Dataset":
        st.subheader("🔍 Cari Siswa di Dataset")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_method = st.radio(
                "Pilih siswa berdasarkan:",
                ["ID Siswa", "Nama"],
                horizontal=True
            )

        if search_method == "ID Siswa":
            student_id = st.number_input(
                "Masukkan ID Siswa",
                min_value=int(df["id"].min()),
                max_value=int(df["id"].max()),
                value=1, step=1
            )
            student_row = df[df["id"] == student_id]
        else:
            df["full_name"] = df["first_name"] + " " + df["last_name"]
            selected_name = st.selectbox(
                "Pilih siswa",
                options=df["full_name"].sort_values().tolist()
            )
            student_row = df[df["full_name"] == selected_name]
            student_id = student_row["id"].values[0]

        if student_row.empty:
            st.warning("Siswa tidak ditemukan.")
            return

        sr = student_row.iloc[0]

        # Tampilkan profil
        st.divider()
        st.markdown(f"### 👤 Profil Siswa: {sr['first_name']} {sr['last_name']}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**ID**: {sr['id']}")
            st.markdown(f"**Gender**: {sr['gender']}")
            st.markdown(f"**Email**: {sr['email']}")
        with c2:
            st.markdown(f"**Part-time job**: {'Ya' if sr['part_time_job'] else 'Tidak'}")
            st.markdown(f"**Ekskul**: {'Ya' if sr['extracurricular_activities'] else 'Tidak'}")
            st.markdown(f"**Absen (hari)**: {sr['absence_days']}")
        with c3:
            st.markdown(f"**Jam belajar/minggu**: {sr['weekly_self_study_hours']}")
            st.markdown(f"**🎯 Cita-cita asli**: `{sr['career_aspiration']}`")

        # Tampilkan nilai-nilai
        st.markdown("#### 📊 Nilai Mata Pelajaran")
        score_data = {
            "Matematika": sr["math_score"],
            "Sejarah": sr["history_score"],
            "Fisika": sr["physics_score"],
            "Kimia": sr["chemistry_score"],
            "Biologi": sr["biology_score"],
            "Bahasa Inggris": sr["english_score"],
            "Geografi": sr["geography_score"]
        }
        st.bar_chart(pd.DataFrame.from_dict(score_data, orient="index", columns=["Nilai"]))

        # Rekomendasi
        st.divider()
        st.markdown("### ✨ Rekomendasi Karier")

        top_n = st.slider("Jumlah rekomendasi", 3, 10, 5)

        if student_id in df_clean["id"].values:
            student_pos = df_clean.index[df_clean["id"] == student_id][0]
            rec = recommend_career_cbf_for_existing(student_pos, df_clean, sim_matrix, top_n=top_n)

            # Tampilkan dengan styling
            for i, row in rec.iterrows():
                is_match = row["Karier"] == sr["career_aspiration"]
                emoji = "🎯" if is_match else f"{i+1}."
                bar_width = int(row["Kepercayaan (%)"] * 5)  # max ~250
                bar = "█" * min(bar_width, 50)

                if is_match:
                    st.success(f"{emoji} **{row['Karier']}** — Kepercayaan: {row['Kepercayaan (%)']}% (cita-cita asli!) {bar}")
                else:
                    st.markdown(f"{emoji} **{row['Karier']}** — Kepercayaan: {row['Kepercayaan (%)']}% {bar}")

            st.dataframe(rec, use_container_width=True, hide_index=True)
        else:
            st.warning("Siswa ini punya cita-cita 'Unknown', tidak bisa direkomendasi.")

    # ====== MODE: INPUT BARU ======
    elif mode == "✏️ Input Profil Siswa Baru":
        st.subheader("✏️ Input Profil Siswa Baru")
        st.markdown("Masukkan profil siswa untuk mendapat rekomendasi karier:")

        with st.form("new_student_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 👤 Profil")
                gender = st.radio("Gender", ["male", "female"], horizontal=True)
                part_time = st.radio("Punya Part-Time Job?", ["Tidak", "Ya"], horizontal=True)
                ekskul = st.radio("Ikut Ekstrakurikuler?", ["Tidak", "Ya"], horizontal=True)
                absence = st.slider("Hari Absen", 0, 10, 3)
                study_hours = st.slider("Jam Belajar Mandiri / Minggu", 0, 50, 15)

            with col2:
                st.markdown("#### 📚 Nilai Mata Pelajaran (0-100)")
                math = st.slider("Matematika", 50, 100, 75)
                history = st.slider("Sejarah", 50, 100, 75)
                physics = st.slider("Fisika", 50, 100, 75)
                chemistry = st.slider("Kimia", 50, 100, 75)
                biology = st.slider("Biologi", 50, 100, 75)
                english = st.slider("Bahasa Inggris", 50, 100, 75)
                geography = st.slider("Geografi", 50, 100, 75)

            top_n_new = st.slider("Jumlah rekomendasi", 3, 10, 5)
            submitted = st.form_submit_button("🎯 Dapatkan Rekomendasi Karier", type="primary", use_container_width=True)

        if submitted:
            profile = [
                1 if gender == "female" else 0,
                1 if part_time == "Ya" else 0,
                absence,
                1 if ekskul == "Ya" else 0,
                study_hours,
                math, history, physics, chemistry, biology, english, geography
            ]

            st.divider()
            st.markdown("### ✨ Hasil Rekomendasi")

            rec = recommend_career_for_new_student(
                profile, df_clean, features_scaled, scaler, feature_cols,
                top_n=top_n_new
            )

            # Tampilkan top-1 sebagai highlight
            top_career = rec.iloc[0]["Karier"]
            st.success(f"🎯 **Rekomendasi Utama**: `{top_career}` (kepercayaan {rec.iloc[0]['Kepercayaan (%)']}%)")

            # Tampilkan semua dengan progress bar
            st.markdown("#### Top-{} Rekomendasi Lengkap".format(top_n_new))
            for i, row in rec.iterrows():
                col_a, col_b = st.columns([3, 7])
                col_a.markdown(f"**{i+1}. {row['Karier']}**")
                col_b.progress(min(row["Kepercayaan (%)"] / 100, 1.0),
                              text=f"Kepercayaan: {row['Kepercayaan (%)']}%")

            # Detail tabel
            with st.expander("📋 Lihat detail tabel"):
                st.dataframe(rec, use_container_width=True, hide_index=True)

    # ====== MODE: EKSPLORASI ======
    elif mode == "📊 Eksplorasi Dataset":
        st.subheader("📊 Eksplorasi Dataset")

        tab1, tab2, tab3 = st.tabs(["📈 Statistik", "🎯 Distribusi Karier", "📚 Nilai Mata Pelajaran"])

        with tab1:
            st.markdown("### Statistik Deskriptif")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Siswa", len(df_clean))
            col2.metric("Total Karier", df_clean["career_aspiration"].nunique())
            col3.metric("Rata-rata Math", f"{df_clean['math_score'].mean():.1f}")
            col4.metric("Rata-rata Study Hours", f"{df_clean['weekly_self_study_hours'].mean():.1f}")

            score_cols = ["math_score", "history_score", "physics_score", "chemistry_score",
                          "biology_score", "english_score", "geography_score"]
            st.markdown("### Statistik Nilai")
            st.dataframe(df_clean[score_cols].describe().round(2), use_container_width=True)

        with tab2:
            st.markdown("### Distribusi Cita-cita Karier")
            career_counts = df_clean["career_aspiration"].value_counts()

            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart(career_counts)
            with col2:
                st.dataframe(
                    career_counts.reset_index().rename(columns={"index": "Karier", "career_aspiration": "Karier", "count": "Jumlah"}),
                    use_container_width=True, hide_index=True
                )

        with tab3:
            st.markdown("### Distribusi Nilai per Mata Pelajaran")
            score_cols = ["math_score", "history_score", "physics_score", "chemistry_score",
                          "biology_score", "english_score", "geography_score"]

            chosen = st.selectbox("Pilih mata pelajaran", score_cols)
            st.bar_chart(df_clean[chosen].value_counts().sort_index())

            st.markdown("### Korelasi Antar Mata Pelajaran")
            st.dataframe(df_clean[score_cols].corr().round(3), use_container_width=True)


if __name__ == "__main__":
    main()
