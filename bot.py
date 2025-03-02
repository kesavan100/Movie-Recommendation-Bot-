import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# 🎬 Page Config
st.set_page_config(page_title="🎥 Tamil Movie Bot", page_icon="🎬", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Tamil_movies_dataset.csv")

movies_df = load_data()
movies_df.columns = movies_df.columns.str.lower()  # Ensure column names are lowercase

# Preprocess data
X = movies_df[['genre', 'year']]
y = movies_df['rating']
encoder = OneHotEncoder(handle_unknown='ignore')
X_genre_encoded = encoder.fit_transform(X[['genre']]).toarray()
X_encoded = pd.DataFrame(X_genre_encoded, columns=encoder.get_feature_names_out(['genre']))
X_encoded = pd.concat([X_encoded, X['year'].reset_index(drop=True)], axis=1)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Function to recommend movies
def recommend_movies(primary_genre, min_rating, year):
    X_all_genre_encoded = encoder.transform(movies_df[['genre']]).toarray()
    X_all_encoded = pd.DataFrame(X_all_genre_encoded, columns=encoder.get_feature_names_out(['genre']))
    X_all_encoded = pd.concat([X_all_encoded, movies_df['year'].reset_index(drop=True)], axis=1)

    movies_df['predictedrating'] = regressor.predict(X_all_encoded)

    recommendations = movies_df[
        (movies_df['genre'].str.contains(primary_genre, case=False, na=False)) & 
        (movies_df['predictedrating'] >= min_rating) & 
        (movies_df['year'] >= year)
    ].sort_values(by='predictedrating', ascending=False)

    if not recommendations.empty:
        return recommendations[['moviename', 'genre', 'predictedrating', 'year']].reset_index(drop=True)
    return pd.DataFrame()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "step" not in st.session_state:
    st.session_state["step"] = 1
    st.session_state["primary_genre"] = None
    st.session_state["min_rating"] = None
    st.session_state["year"] = None
if "last_input" not in st.session_state:
    st.session_state["last_input"] = ""

# Display chat history
for message in st.session_state["messages"]:
    st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("💬 Type your message...")

if user_input and user_input != st.session_state["last_input"]:  # Ensure input is processed once
    st.session_state["last_input"] = user_input
    st.session_state["messages"].append({"role": "user", "content": f"👤 {user_input}"})

    # Handle conversation steps
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
            st.session_state["step"] = 4

            recommendations = recommend_movies(
                st.session_state["primary_genre"], st.session_state["min_rating"], st.session_state["year"]
            )

            if not recommendations.empty:
                response = "🤖 🎥 **Here are your recommended movies:**\n\n"
                response += f"🎬 **Movie Name**{' ' * 10}🎭 **Genre**{' ' * 10}⭐ **Rating**{' ' * 6}📅 **Year**\n"
                response += "-" * 60 + "\n"

                for _, row in recommendations.iterrows():
                    response += f"{row['moviename'][:18]:<22}{row['genre'][:18]:<22}{row['predictedrating']:.1f}{' ' * 10}{row['year']}\n"

                response += "\n✨ Type **'restart'** to search again!"
            else:
                response = "🤖 ❌ No movies found! Type 'restart' to try again."

        except ValueError:
            response = "🤖 ❌ Please enter a valid year."

    elif user_input.lower() == "restart":
        st.session_state["step"] = 1
        response = "🤖 🔄 Restarting... 👋 Hi again! What genre of movie are you looking for? 🎭"

    # Append bot response
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Rerun the app to display updated chat
    st.rerun()
