import streamlit as st
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Tamil Movie Chatbot", layout="wide")

# 🎨 Custom CSS for WhatsApp-style chat
st.markdown(
    """
    <style>
        .stChatMessage { padding: 10px; border-radius: 10px; margin: 5px 0; max-width: 80%; }
        .user { background-color: #DCF8C6; align-self: flex-end; }
        .bot { background-color: #EAEAEA; align-self: flex-start; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 Tamil Movie Chatbot 🎬")
st.write("👋 Hey there! Let's find you the perfect Tamil movie! 🎥")

# 🗂 Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Tamil_movies_dataset.csv")

movies_df = load_data()

# 🎭 Preprocess data
X = movies_df[['Genre', 'Year']]
y = movies_df['Rating']
encoder = OneHotEncoder(handle_unknown='ignore')
X_genre_encoded = encoder.fit_transform(X[['Genre']]).toarray()
X_encoded = pd.DataFrame(X_genre_encoded, columns=encoder.get_feature_names_out(['Genre']))
X_encoded = pd.concat([X_encoded, X['Year'].reset_index(drop=True)], axis=1)

# 🧠 Train model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# 🎯 Recommendation Function
def recommend_movies(primary_genre, secondary_genre, min_rating, year):
    X_all_genre_encoded = encoder.transform(movies_df[['Genre']]).toarray()
    X_all_encoded = pd.DataFrame(X_all_genre_encoded, columns=encoder.get_feature_names_out(['Genre']))
    X_all_encoded = pd.concat([X_all_encoded, movies_df['Year'].reset_index(drop=True)], axis=1)

    movies_df['PredictedRating'] = regressor.predict(X_all_encoded)

    recommendations = movies_df[
        (movies_df['Genre'].str.contains(primary_genre, case=False)) & 
        (movies_df['PredictedRating'] >= min_rating) & 
        (movies_df['Year'] >= year)
    ]

    if secondary_genre and secondary_genre.lower() != "none":
        recommendations = recommendations[recommendations['Genre'].str.contains(secondary_genre, case=False)]

    recommendations = recommendations.sort_values(by='PredictedRating', ascending=False)
    return recommendations[['MovieName', 'Genre', 'PredictedRating', 'Year', 'PosterURL', 'IMDbID']].reset_index(drop=True)

# 🔄 Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "step" not in st.session_state:
    st.session_state["step"] = 1
    st.session_state["primary_genre"] = None
    st.session_state["secondary_genre"] = None
    st.session_state["min_rating"] = None
    st.session_state["year"] = None

# 🗨️ Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 🔤 User input
user_input = st.chat_input("Type your response...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process user input
    bot_response = ""
    
    if st.session_state["step"] == 1:
        st.session_state["primary_genre"] = user_input
        st.session_state["step"] = 2
        bot_response = f"🎭 Got it! You like **{user_input}**. Do you want a secondary genre? (Type 'None' to skip)"

    elif st.session_state["step"] == 2:
        st.session_state["secondary_genre"] = user_input
        st.session_state["step"] = 3
        bot_response = "🔢 Great! What's the minimum rating you'd prefer? (0-10)"

    elif st.session_state["step"] == 3:
        try:
            st.session_state["min_rating"] = float(user_input)
            if 0 <= st.session_state["min_rating"] <= 10:
                st.session_state["step"] = 4
                bot_response = "📅 From which year should I suggest movies?"
            else:
                bot_response = "❌ Please enter a rating between 0 and 10."
        except ValueError:
            bot_response = "❌ Please enter a valid number."

    elif st.session_state["step"] == 4:
        try:
            st.session_state["year"] = int(user_input)
            st.session_state["step"] = 5

            # 🔥 Get recommendations
            primary_genre = st.session_state["primary_genre"]
            secondary_genre = st.session_state["secondary_genre"]
            min_rating = st.session_state["min_rating"]
            year = st.session_state["year"]

            recommendations = recommend_movies(primary_genre, secondary_genre, min_rating, year)

            if not recommendations.empty:
                bot_response = "🎥 **Here are your recommended movies:**\n"
                for i, movie in recommendations.iterrows():
                    movie_text = f"📽 **{movie['MovieName']} ({movie['Year']})**\n⭐ {movie['PredictedRating']:.1f}\n[🔗 IMDb](https://www.imdb.com/title/{movie['IMDbID']}/)\n\n"
                    bot_response += movie_text
            else:
                bot_response = "❌ No movies found! Type 'restart' to try again."

        except ValueError:
            bot_response = "❌ Please enter a valid year."

    elif user_input.lower() == "restart":
        st.session_state["step"] = 1
        st.session_state["primary_genre"] = None
        st.session_state["secondary_genre"] = None
        st.session_state["min_rating"] = None
        st.session_state["year"] = None
        bot_response = "🔄 Restarting... 👋 Hi again! Let's find your perfect movie."

    # Display chatbot response in chat
    st.session_state["messages"].append({"role": "bot", "content": bot_response})

    with st.chat_message("bot"):
        st.markdown(bot_response)
