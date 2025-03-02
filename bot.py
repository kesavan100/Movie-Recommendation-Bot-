import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# 🎬 Page Config (Reduced screen width)
st.set_page_config(page_title="🎥 Tamil Movie Bot", page_icon="🎬", layout="centered")

# Custom CSS for WhatsApp-style chat layout
st.markdown("""
    <style>
        .chat-container {
            max-width: 600px;
            margin: auto;
        }
        .chat-message {
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: inline-block;
            max-width: 80%;
        }
        .user-message {
            background-color: #dcf8c6;
            text-align: right;
            float: right;
            clear: both;
        }
        .bot-message {
            background-color: #f1f0f0;
            text-align: left;
            float: left;
            clear: both;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Greeting
st.title("🤖 Tamil Movie Recommendation Bot")
st.write("👋 **Hello!** I'm your AI-powered movie assistant. Let’s find the perfect Tamil movie for you!")
st.write("🎥 **Enter a Genre to Get Recommendations!**")

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

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "step" not in st.session_state:
    st.session_state["step"] = 1
    st.session_state["primary_genre"] = None
    st.session_state["min_rating"] = None
    st.session_state["year"] = None
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = pd.DataFrame()
if "movie_index" not in st.session_state:
    st.session_state["movie_index"] = 0

# Display chat history (Styled as WhatsApp chat)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state["messages"]:
    role_class = "user-message" if message["role"] == "user" else "bot-message"
    prefix = "✨" if message["role"] == "user" else "🤖 "
    st.markdown(f'<div class="chat-message {role_class}">{prefix}{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Custom chat input
user_input = st.chat_input("💬 Type your message...")

if user_input:
    # Append user message with ✨ emoji
    st.session_state["messages"].append({"role": "user", "content": f"✨ {user_input}"})

    # Default response to avoid undefined variable error
    response = "🤖 I'm not sure what you mean. Please try again!"

    # Process chatbot response
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
            st.session_state["recommendations"] = recommend_movies(
                st.session_state["primary_genre"], st.session_state["min_rating"], st.session_state["year"]
            )
            st.session_state["movie_index"] = 0
            st.session_state["step"] = 4
        except ValueError:
            response = "🤖 ❌ Please enter a valid year."

    elif st.session_state["step"] == 4:
        if user_input.lower() == "exit":
            st.session_state["step"] = 1
            response = "🤖 👋 Goodbye! Type again to start a new search."
        elif user_input.lower() == "more":
            index = st.session_state["movie_index"]
            recommendations = st.session_state["recommendations"]

            if recommendations.empty:
                response = "🤖 ❌ No movies found! Type 'restart' to try again."
            else:
                response = "🤖 🎥 **Here are more movies for you:**\n\n"

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
