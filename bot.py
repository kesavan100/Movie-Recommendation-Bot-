import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# 🎬 Page Config
st.set_page_config(page_title="🎥 Tamil Movie Bot", page_icon="🎬", layout="centered")

# Load dataset & cache it
@st.cache_data
def load_data():
    df = pd.read_csv("Tamil_movies_dataset.csv")
    df.columns = df.columns.str.lower()  # Ensure lowercase column names
    return df

movies_df = load_data()

# Preprocess data
@st.cache_data
def train_model(df):
    X = df[['genre', 'year']]
    y = df['rating']
    
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_genre_encoded = encoder.fit_transform(X[['genre']]).toarray()
    
    X_encoded = pd.DataFrame(X_genre_encoded, columns=encoder.get_feature_names_out(['genre']))
    X_encoded = pd.concat([X_encoded, X['year'].reset_index(drop=True)], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict ratings and store them
    X_all_genre_encoded = encoder.transform(df[['genre']]).toarray()
    X_all_encoded = pd.DataFrame(X_all_genre_encoded, columns=encoder.get_feature_names_out(['genre']))
    X_all_encoded = pd.concat([X_all_encoded, df['year'].reset_index(drop=True)], axis=1)

    df["predictedrating"] = model.predict(X_all_encoded)

    return df, encoder

movies_df, encoder = train_model(movies_df)

# Function to recommend movies (cached)
@st.cache_data
def recommend_movies(primary_genre, min_rating, year):
    filtered_df = movies_df[movies_df['genre'].str.contains(primary_genre, case=False, na=False)]
    recommendations = filtered_df.query("predictedrating >= @min_rating & year >= @year") \
                                .sort_values(by="predictedrating", ascending=False)[['moviename', 'genre', 'predictedrating', 'year']]
    
    return recommendations.reset_index(drop=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "step" not in st.session_state:
    st.session_state["step"] = 1
    st.session_state["primary_genre"] = None
    st.session_state["min_rating"] = None
    st.session_state["year"] = None
    st.session_state["recommendations"] = None
    st.session_state["movie_index"] = 0

# Display chat history
for message in st.session_state["messages"]:
    st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

# Custom chat input
user_input = st.chat_input("💬 Type your message...")

if user_input:
    # Append user message only once
    st.session_state["messages"].append({"role": "user", "content": user_input})

    response = "🤖 I didn't understand that. Try again."

    if st.session_state["step"] == 1:
        st.session_state["primary_genre"] = user_input.lower()
        st.session_state["step"] = 2
        response = "🤖 🎭 Got it! What minimum rating do you prefer? (0-10) ⭐"

    elif st.session_state["step"] == 2:
        try:
            rating = float(user_input)
            if 0 <= rating <= 10:
                st.session_state["min_rating"] = rating
                st.session_state["step"] = 3
                response = "🤖 📅 From which year should I suggest movies? 🎬"
            else:
                response = "🤖 ❌ Please enter a rating between 0 and 10."
        except ValueError:
            response = "🤖 ❌ Please enter a valid number."

    elif st.session_state["step"] == 3:
        try:
            year = int(user_input)
            st.session_state["year"] = year
            st.session_state["recommendations"] = recommend_movies(st.session_state["primary_genre"], st.session_state["min_rating"], st.session_state["year"])
            st.session_state["movie_index"] = 0
            st.session_state["step"] = 4
        except ValueError:
            response = "🤖 ❌ Please enter a valid year."

    if st.session_state["step"] == 4:
        if user_input.lower() == "exit":
            st.session_state["step"] = 1
            response = "🤖 👋 Goodbye! Type again to start a new search."
        elif user_input.lower() == "more" or st.session_state["movie_index"] == 0:
            recommendations = st.session_state["recommendations"]
            index = st.session_state["movie_index"]

            if recommendations.empty:
                response = "🤖 ❌ No movies found! Type 'restart' to try again."
            else:
                response = "🤖 🎥 **Here are your recommended movies:**\n\n"

                for i in range(index, min(index + 5, len(recommendations))):
                    row = recommendations.iloc[i]
                    response += (
                        f"🎬 **{row['moviename']}**\n"
                        f"🎭 Genre: {row['genre']}\n"
                        f"⭐ Rating: {row['predictedrating']:.1f}\n"
                        f"📅 Year: {row['year']}\n\n"
                    )

                st.session_state["movie_index"] += 5

                if st.session_state["movie_index"] >= len(recommendations):
                    response += "✨ No more movies left! Type 'restart' to start over."
                else:
                    response += "✨ Type 'more' to see more movies or 'exit' to stop."

    # Append bot response
    st.session_state["messages"].append({"role": "assistant", "content": response})
