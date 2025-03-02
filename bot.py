import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# 🎬 Title with Style
st.markdown(
    """
    <h1 style='text-align: center; color: #FFD700;'>🎥 Tamil Movie Chatbot</h1>
    """,
    unsafe_allow_html=True,
)

# Background Styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: white;
    }
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
    }
    .user {
        background-color: #0f62fe;
        color: white;
        align-self: flex-end;
    }
    .assistant {
        background-color: #262626;
        color: white;
    }
    .movie-card {
        background-color: #262626;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        (movies_df['Genre'].str.lower().str.contains(primary_genre.lower(), case=False)) & 
        (movies_df['PredictedRating'] >= min_rating) & 
        (movies_df['Year'] >= year)
    ]
    
    recommendations = recommendations.sort_values(by='PredictedRating', ascending=False)
    return recommendations[['MovieName', 'Genre', 'PredictedRating', 'Year', 'PosterURL']].reset_index(drop=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "👋 Hello! I am a **Movie Recommendation Bot**. 🎬 I can help you find great Tamil movies!"},
        {"role": "assistant", "content": "🎭 Which **genre** of movie do you want recommendations for?"}
    ]
    st.session_state.step = 1
    st.session_state.primary_genre = None
    st.session_state.min_rating = None
    st.session_state.year = None

# Display chat history with bubbles
for message in st.session_state.chat_history:
    css_class = "user" if message["role"] == "user" else "assistant"
    st.markdown(f"<div class='stChatMessage {css_class}'>{message['content']}</div>", unsafe_allow_html=True)

# Chatbot Interaction
user_input = st.chat_input("Type your response here...")

if user_input:
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    if st.session_state.step == 1:
        st.session_state.primary_genre = user_input
        st.session_state.step = 2
        bot_message = f"🎭 Got it! You like **{user_input}**. What's the **minimum rating** you prefer? (0-10)"
    
    elif st.session_state.step == 2:
        try:
            st.session_state.min_rating = float(user_input)
            if 0 <= st.session_state.min_rating <= 10:
                st.session_state.step = 3
                bot_message = "📅 From which **year** should I suggest movies?"
            else:
                bot_message = "❌ Please enter a rating between **0 and 10**."
        except ValueError:
            bot_message = "❌ Please enter a **valid number**."

    elif st.session_state.step == 3:
        try:
            st.session_state.year = int(user_input)
            st.session_state.step = 4

            # Get recommendations
            primary_genre = st.session_state.primary_genre
            min_rating = st.session_state.min_rating
            year = st.session_state.year
            
            recommendations = recommend_movies(primary_genre, min_rating, year)

            if not recommendations.empty:
                bot_message = "🎥 **Here are your recommended movies:**"
                for _, movie in recommendations.iterrows():
                    if pd.notna(movie.PosterURL):  # Check if poster URL is available
                        st.image(movie.PosterURL, width=150)
                    
                    st.markdown(
                        f"""
                        <div class="movie-card">
                        📽 <b>{movie.MovieName} ({movie.Year})</b> - ⭐ {movie.PredictedRating:.1f} | 🎭 {movie.Genre}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                bot_message += "\n✨ Type '**restart**' to search again!"
            else:
                bot_message = "❌ No movies found! Type '**restart**' to try again."
        
        except ValueError:
            bot_message = "❌ Please enter a **valid year**."

    elif user_input.lower() == "restart":
        st.session_state.chat_history = [
            {"role": "assistant", "content": "🔄 Restarting... 👋 Hello again! I am a **Movie Recommendation Bot**."},
            {"role": "assistant", "content": "🎭 Which **genre** of movie do you want recommendations for?"}
        ]
        st.session_state.step = 1
        st.session_state.primary_genre = None
        st.session_state.min_rating = None
        st.session_state.year = None
        bot_message = None  # No need to add another message

    # Append bot response to chat history
    if bot_message:
        st.session_state.chat_history.append({"role": "assistant", "content": bot_message})
    
    # Refresh the page with updated chat history
    st.rerun()
