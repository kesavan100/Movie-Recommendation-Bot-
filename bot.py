import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# 🎬 Title
st.title("🎥 Tamil Movie Chatbot")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Tamil_movies_dataset.csv")

movies_df = load_data()

# Preprocess data
X = movies_df[['Genre', 'Year']]
y = movies_df['Rating']
encoder = OneHotEncoder(handle_unknown='ignore')
X_genre_encoded = encoder.fit_transform(X[['Genre']]).toarray()
X_encoded = pd.DataFrame(X_genre_encoded, columns=encoder.get_feature_names_out(['Genre']))
X_encoded = pd.concat([X_encoded, X['Year'].reset_index(drop=True)], axis=1)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Function to recommend movies
def recommend_movies(primary_genre, min_rating, year):
    X_all_genre_encoded = encoder.transform(movies_df[['Genre']]).toarray()
    X_all_encoded = pd.DataFrame(X_all_genre_encoded, columns=encoder.get_feature_names_out(['Genre']))
    X_all_encoded = pd.concat([X_all_encoded, movies_df['Year'].reset_index(drop=True)], axis=1)
    
    movies_df['PredictedRating'] = regressor.predict(X_all_encoded)

    recommendations = movies_df[
        (movies_df['Genre'].str.contains(primary_genre, case=False)) & 
        (movies_df['PredictedRating'] >= min_rating) & 
        (movies_df['Year'] >= year)
    ]
    
    recommendations = recommendations.sort_values(by='PredictedRating', ascending=False)
    return recommendations[['MovieName', 'Genre', 'PredictedRating', 'Year']].reset_index(drop=True)

# Initialize session state
if "step" not in st.session_state:
    st.session_state["step"] = 1
    st.session_state["primary_genre"] = None
    st.session_state["min_rating"] = None
    st.session_state["year"] = None

# Chatbot Interaction
st.write("👋 Hi! I'm your movie chatbot. Let's find the best Tamil movies for you!")
st.write("🎥 Enter Genre of the Movie")
user_input = st.chat_input("Type your response here...")

if user_input:
    if st.session_state["step"] == 1:
        st.session_state["primary_genre"] = user_input
        st.session_state["step"] = 2
        st.write(f"🎭 Got it! You like **{user_input}**. What's the minimum rating you prefer? (0-10)")
    
    elif st.session_state["step"] == 2:
        try:
            st.session_state["min_rating"] = float(user_input)
            if 0 <= st.session_state["min_rating"] <= 10:
                st.session_state["step"] = 3
                st.write("📅 From which year should I suggest movies?")
            else:
                st.write("❌ Please enter a rating between 0 and 10.")
        except ValueError:
            st.write("❌ Please enter a valid number.")
    
    elif st.session_state["step"] == 3:
        try:
            st.session_state["year"] = int(user_input)
            st.session_state["step"] = 4
            
            # Get recommendations
            primary_genre = st.session_state["primary_genre"]
            min_rating = st.session_state["min_rating"]
            year = st.session_state["year"]
            
            recommendations = recommend_movies(primary_genre, min_rating, year)
            
            if not recommendations.empty:
                st.write("🎥 **Here are your recommended movies:**")
                for _, movie in recommendations.iterrows():
                    st.write(f"📽 **{movie.MovieName} ({movie.Year})** - ⭐ {movie.PredictedRating:.1f} | 🎭 {movie.Genre}")
                st.write("✨ Type 'restart' to search again!")
            else:
                st.write("❌ No movies found! Type 'restart' to try again.")
            
        except ValueError:
            st.write("❌ Please enter a valid year.")

    elif user_input.lower() == "restart":
        st.session_state["step"] = 1
        st.session_state["primary_genre"] = None
        st.session_state["min_rating"] = None
        st.session_state["year"] = None
        st.write("🔄 Restarting... 👋 Hi again! Let's find your perfect movie.")
