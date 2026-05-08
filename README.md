# 🎓 Career Recommendation System for Students

## 🎯 Overview

Choosing the right career path is one of the most important decisions a student makes. However, with countless career options available, many students feel overwhelmed and uncertain about which direction aligns best with their abilities and interests.

This project provides a **data-driven solution** by analyzing students' academic performance, study patterns, and extracurricular involvement to recommend the most suitable career aspirations using machine learning.

### Why This Matters

- **11% of students** in the original dataset (223 out of 2,000) had no clear career direction
- Career mismatch leads to lower job satisfaction and reduced productivity
- Data-driven recommendations can complement traditional guidance counseling

---

## ✨ Features

🏠 **Interactive Dashboard** — Get an overview of dataset statistics and career distribution

🔍 **Existing Student Lookup** — Browse students from the database and view their personalized recommendations

✏️ **Custom Profile Input** — Enter any student profile manually and receive instant career recommendations

📊 **Data Exploration** — Visualize career distributions, score patterns, and inter-subject correlations

🎯 **Confidence Scoring** — Each recommendation comes with a confidence percentage to support decision-making

---

## 🚀 Live Demo

🔗 **Try the app**: https://career-recommender-for-students-by-gita.streamlit.app

---

## 🤖 How It Works

### Algorithm: Content-Based Filtering with Cosine Similarity

The system follows a **3-step approach**:

```
Student Profile → Feature Extraction → Similarity Computation → Career Recommendations
```

1. **Feature Engineering** — 12 normalized features extracted from each student's profile:
   - 7 academic scores (Math, Physics, Chemistry, Biology, History, English, Geography)
   - 5 behavioral attributes (study hours, attendance, extracurricular activities, part-time job, gender)

2. **Similarity Computation** — Cosine similarity is calculated between the target student and all 1,777 students in the database

3. **Weighted Voting** — Top-50 most similar students are identified, and their career aspirations are aggregated using similarity-weighted voting to produce ranked recommendations

---

## 📊 Dataset

The system is trained on a publicly available student dataset containing **2,000 records** with **17 attributes**:

- **Demographics**: gender, ID
- **Behavioral**: part-time job status, extracurricular participation, absence days, weekly study hours
- **Academic**: scores in 7 core subjects (0-100 scale)
- **Target**: career aspiration (17 distinct categories)

---

## 🔮 Future Enhancements

- [ ] Implement **Collaborative Filtering** with neural network embeddings
- [ ] Build a **Hybrid Recommender** combining content-based and collaborative approaches
- [ ] Add **explainability features** to show why a career was recommended
- [ ] Integrate **personality assessments** (e.g., MBTI, RIASEC) for richer profiling
- [ ] Support **multi-language interface** (English, Indonesian)
- [ ] Add **historical job market data** to suggest in-demand careers
- [ ] Build a **mobile-responsive PWA** version

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

If you use this project as a reference, please consider citing it:

```bibtex
@software{career_recommender_2026,
  author = {Gita Ramadhani W.S},
  title = {Career Recommendation System for Students},
  year = {2026},
  url = {https://github.com/G1taaRamadhan1/career-recommender-using-streamlit.git}
}
```

---

## 👤 Author

**Gita Ramadhani W.S**

- 🐙 GitHub: [@G1taaRamadhan1](https://github.com/G1taaRamadhan1)
- 💼 LinkedIn: [Gita Ramadhani W.S](https://linkedin.com/in/gitaramadhaniws)
- 📧 Email: workwith.gitaramadhani@gmail.com

---

## ⭐ Show Your Support

If this project helped you, please consider giving it a ⭐ on GitHub!

---

