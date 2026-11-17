import os
import time
import requests
import pandas as pd
import nltk
import plotly.express as px
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page setup
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# Download VADER lexicon quietly for sentiment analysis
@st.cache_resource
def load_sentiment_analyzer():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_sentiment_analyzer()

# Helper to fetch from TMDB if CSV is missing
def fetch_tmdb_data():
    token = st.secrets.get("TMDB_READ_ACCESS_TOKEN", os.getenv("TMDB_READ_ACCESS_TOKEN", ""))
    if not token:
        return pd.DataFrame()
    headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}
    movie_list = []
    for page in range(1, 6):
        r = requests.get(
            "https://api.themoviedb.org/3/movie/popular",
            headers=headers,
            params={"language": "en-US", "page": page},
            timeout=15,
        )
        if r.status_code == 200:
            movie_list.extend(r.json().get("results", []))

    rows = []
    for m in movie_list:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{m['id']}",
            headers=headers,
            params={"language": "en-US", "append_to_response": "credits,keywords,release_dates"},
            timeout=15,
        )
        if r.status_code == 200:
            d = r.json()
            crew = d.get("credits", {}).get("crew", [])
            cast = d.get("credits", {}).get("cast", [])
            kw = d.get("keywords", {}).get("keywords", [])
            cert = "Not available"
            for c in d.get("release_dates", {}).get("results", []):
                if c.get("iso_3166_1") == "US":
                    certs = [x["certification"] for x in c.get("release_dates", []) if x.get("certification")]
                    if certs:
                        cert = certs[0]
            rows.append({
                "id": m["id"],
                "title": d.get("title"),
                "year": str(d.get("release_date", ""))[:4],
                "overview": d.get("overview", ""),
                "genres": ", ".join(g["name"] for g in d.get("genres", [])),
                "runtime": d.get("runtime", 0),
                "rating": d.get("vote_average", 0),
                "vote_count": d.get("vote_count", 0),
                "language": d.get("original_language", ""),
                "actors": ", ".join(p["name"] for p in cast[:5]),
                "directors": ", ".join(p["name"] for p in crew if p.get("job") == "Director"),
                "keywords": ", ".join(k["name"] for k in kw),
                "certification": cert,
                "budget": d.get("budget", 0),
                "revenue": d.get("revenue", 0),
                "poster": f"https://image.tmdb.org/t/p/w500{d['poster_path']}" if d.get("poster_path") else "",
            })
        time.sleep(0.1)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv("cinematch_movies.csv", index=False)
    return df

# Load & enrich movie database
@st.cache_data
def load_movies():
    if os.path.exists("cinematch_movies.csv"):
        df = pd.read_csv("cinematch_movies.csv")
    else:
        df = fetch_tmdb_data()

    if df.empty:
        return df

    text_cols = ["title", "overview", "genres", "actors", "directors", "keywords", "language", "certification", "poster"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    for col in ["year", "rating", "vote_count", "runtime", "budget", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # NLP Sentiment Analysis
    df["sentiment_score"] = df["overview"].apply(lambda t: sia.polarity_scores(t)["compound"])
    df["sentiment"] = df["sentiment_score"].apply(
        lambda s: "Positive / Uplifting" if s >= 0.05 else ("Dark / Intense" if s <= -0.05 else "Neutral")
    )

    df["combined_features"] = (
        df["overview"] + " " + df["genres"] + " " + df["keywords"] + " " + df["actors"] + " " + df["directors"]
    )
    return df

movies = load_movies()

if movies.empty:
    st.error("Movie database not found. Upload `cinematch_movies.csv` to your GitHub repo or add `TMDB_READ_ACCESS_TOKEN` in Streamlit Secrets.")
    st.stop()

# Build TF-IDF Recommendation Model
@st.cache_resource
def build_model(features):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform(features)
    return cosine_similarity(matrix)

cosine_sim = build_model(movies["combined_features"].tolist())

# Initialize Watchlist in Session State
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# Header
st.title("🎬 CineMatch")
st.subheader("AI-Powered Movie Discovery & Hybrid Recommendation Engine")

# Sidebar Filters (All Core Filtering Requirements)
st.sidebar.header("🔎 Core Filters")
search = st.sidebar.text_input("Search movie title")

genre_options = sorted({g.strip() for gs in movies["genres"] for g in gs.split(",") if g.strip()})
selected_genres = st.sidebar.multiselect("Choose genres", genre_options)
genre_logic = st.sidebar.radio("Genre match logic", ["AND (Match all)", "OR (Match any)"], horizontal=True)

min_y, max_y = int(movies[movies["year"] > 0]["year"].min()), int(movies["year"].max())
year_range = st.sidebar.slider("Release year range", min_y, max_y, (min_y, max_y))

min_rating = st.sidebar.slider("Minimum TMDB rating", 0.0, 10.0, 0.0, 0.5)
min_votes = st.sidebar.number_input("Minimum vote count", min_value=0, value=0, step=50)
max_runtime = st.sidebar.number_input("Maximum runtime (mins, 0 = any)", min_value=0, value=0, step=15)

selected_language = st.sidebar.selectbox("Language", ["All"] + sorted(movies["language"].unique().tolist()))
selected_cert = st.sidebar.selectbox("Certification", ["All"] + sorted([x for x in movies["certification"].unique() if x]))
selected_mood = st.sidebar.selectbox("NLP Plot Sentiment / Mood", ["All", "Positive / Uplifting", "Neutral", "Dark / Intense"])

actor_search = st.sidebar.text_input("Actor name")
director_search = st.sidebar.text_input("Director name")

# Apply Filters
filtered = movies.copy()
if search:
    filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]
if selected_genres:
    if genre_logic.startswith("AND"):
        filtered = filtered[filtered["genres"].apply(lambda v: all(g in v for g in selected_genres))]
    else:
        filtered = filtered[filtered["genres"].apply(lambda v: any(g in v for g in selected_genres))]

filtered = filtered[
    (filtered["year"] >= year_range[0]) &
    (filtered["year"] <= year_range[1]) &
    (filtered["rating"] >= min_rating) &
    (filtered["vote_count"] >= min_votes)
]
if max_runtime > 0:
    filtered = filtered[filtered["runtime"] <= max_runtime]
if selected_language != "All":
    filtered = filtered[filtered["language"] == selected_language]
if selected_cert != "All":
    filtered = filtered[filtered["certification"] == selected_cert]
if selected_mood != "All":
    filtered = filtered[filtered["sentiment"] == selected_mood]
if actor_search:
    filtered = filtered[filtered["actors"].str.contains(actor_search, case=False, na=False)]
if director_search:
    filtered = filtered[filtered["directors"].str.contains(director_search, case=False, na=False)]

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🍿 Explore & Filter",
    "✨ Hybrid Recommendations",
    "📌 My Watchlist",
    "⚖️ Compare Movies",
    "📊 Trends & Analytics",
])

# TAB 1: Explore Movies
with tab1:
    st.write(f"Showing **{len(filtered)}** matching movies")
    if filtered.empty:
        st.warning("No movies match your filter combination. Try relaxing one or more sidebar filters.")
    else:
        cols = st.columns(3)
        for i, (_, m) in enumerate(filtered.iterrows()):
            with cols[i % 3]:
                if m["poster"].startswith("http"):
                    st.image(m["poster"], use_container_width=True)
                st.subheader(m["title"])
                st.write(f"📅 **{int(m['year'])}** | ⭐ **{m['rating']}/10** ({int(m['vote_count'])} votes)")
                st.write(f"🎭 {m['genres']} | ⏱️ {int(m['runtime'])}m | 🔞 {m['certification']}")
                st.write(f"🧠 **NLP Tone:** {m['sentiment']} ({m['sentiment_score']:+.2f})")
                st.caption(m["overview"] or "No description available.")
                if st.button("➕ Add to Watchlist", key=f"wl_{m['id']}"):
                    if m["title"] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(m["title"])
                        st.toast(f"Added {m['title']} to Watchlist!")
                st.divider()

# TAB 2: Hybrid Recommendations (Content + Sentiment)
with tab2:
    st.header("✨ Hybrid Recommendation Engine")
    st.write("Combines **TF-IDF Content Similarity** (plot, genres, keywords, cast, director) with **VADER Sentiment Alignment**.")
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        selected_movie = st.selectbox("Select a movie you enjoyed", sorted(movies["title"].unique()))
    with col_b:
        sentiment_weight = st.slider("Sentiment match weight", 0.0, 0.5, 0.25, 0.05)
    with col_c:
        num_recs = st.slider("Number of results", 1, 10, 6)

    if st.button("Generate Recommendations", type="primary"):
        idx = movies[movies["title"] == selected_movie].index[0]
        base_sentiment = movies.loc[idx, "sentiment_score"]

        hybrid_scores = []
        for i, content_score in enumerate(cosine_sim[idx]):
            if i == idx:
                continue
            sent_diff = abs(base_sentiment - movies.loc[i, "sentiment_score"]) / 2.0
            sent_sim = 1.0 - sent_diff
            final_score = (1 - sentiment_weight) * content_score + sentiment_weight * sent_sim
            hybrid_scores.append((i, final_score, content_score, sent_sim))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        rec_cols = st.columns(3)
        for i, (m_idx, total_s, cont_s, sent_s) in enumerate(hybrid_scores[:num_recs]):
            m = movies.iloc[m_idx]
            with rec_cols[i % 3]:
                if m["poster"].startswith("http"):
                    st.image(m["poster"], use_container_width=True)
                st.subheader(m["title"])
                st.write(f"⭐ **{m['rating']}/10** | 🎭 {m['genres']}")
                st.success(f"**Hybrid Match: {total_s * 100:.1f}%**")
                st.caption(f"Content Similarity: {cont_s * 100:.1f}% | Mood Alignment: {sent_s * 100:.1f}% ({m['sentiment']})")

# TAB 3: User Watchlist & Profile-Based Recommendations
with tab3:
    st.header("📌 Your Watchlist & Profile Recommendations")
    if not st.session_state.watchlist:
        st.info("Your watchlist is currently empty. Add movies from the **Explore & Filter** tab to get profile-based recommendations!")
    else:
        st.write("**Saved Movies:**", ", ".join(st.session_state.watchlist))
        if st.button("Clear Watchlist"):
            st.session_state.watchlist = []
            st.rerun()

        wl_indices = movies[movies["title"].isin(st.session_state.watchlist)].index.tolist()
        avg_profile = cosine_sim[wl_indices].mean(axis=0)
        profile_recs = sorted(
            [(i, s) for i, s in enumerate(avg_profile) if i not in wl_indices],
            key=lambda x: x[1],
            reverse=True,
        )[:6]

        st.subheader("🎯 Recommended Based on Your Watchlist Profile")
        wl_cols = st.columns(3)
        for i, (m_idx, score) in enumerate(profile_recs):
            m = movies.iloc[m_idx]
            with wl_cols[i % 3]:
                if m["poster"].startswith("http"):
                    st.image(m["poster"], use_container_width=True)
                st.write(f"**{m['title']}** ({int(m['year'])})")
                st.write(f"⭐ {m['rating']}/10 | Profile Match: **{score * 100:.1f}%**")
                st.caption(m["genres"])

# TAB 4: Comparative Analysis
with tab4:
    st.header("⚖️ Side-by-Side Movie Comparison")
    titles = sorted(movies["title"].unique())
    c1, c2 = st.columns(2)
    m1_title = c1.selectbox("Movie A", titles, index=0)
    m2_title = c2.selectbox("Movie B", titles, index=min(1, len(titles) - 1))

    m1 = movies[movies["title"] == m1_title].iloc[0]
    m2 = movies[movies["title"] == m2_title].iloc[0]

    comp_df = pd.DataFrame({
        "Attribute": ["Release Year", "TMDB Rating", "Vote Count", "Runtime (mins)", "Certification", "NLP Sentiment", "Budget ($)", "Revenue ($)"],
        m1_title: [int(m1["year"]), m1["rating"], int(m1["vote_count"]), int(m1["runtime"]), m1["certification"], f"{m1['sentiment']} ({m1['sentiment_score']:+.2f})", f"${int(m1['budget']):,}", f"${int(m1['revenue']):,}"],
        m2_title: [int(m2["year"]), m2["rating"], int(m2["vote_count"]), int(m2["runtime"]), m2["certification"], f"{m2['sentiment']} ({m2['sentiment_score']:+.2f})", f"${int(m2['budget']):,}", f"${int(m2['revenue']):,}"],
    })
    st.table(comp_df)

# TAB 5: Data Visualizations
with tab5:
    st.header("📊 Catalog Trends & NLP Analytics")
    v1, v2 = st.columns(2)
    with v1:
        fig1 = px.scatter(
            movies,
            x="vote_count",
            y="rating",
            color="sentiment",
            hover_name="title",
            title="TMDB Rating vs. Popularity (Colored by NLP Sentiment)",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with v2:
        sentiment_counts = movies["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig2 = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Plot Sentiment Distribution")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("CineMatch | This product uses the TMDB API but is not endorsed or certified by TMDB.")
