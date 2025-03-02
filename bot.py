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
        .chat-container { max-width: 600px; margin: auto; }
        .chat-message {
            padding: 10px; border-radius: 10px; margin-bottom: 10px;
            display: inline-block; max-width: 80%;
        }
        .user-message { background-color: #dcf8c6; text-align: right; float: right; clear: both; }
        .bot-message { background-color: #f1f0f0; text-align: left; float: left; clear: both; }
    </style>
""", unsafe_allow_html=True)

# Title and Greeting
st.title("🤖 Tamil Movie Recommendation Bot")
st.write("👋 **Hello!** I'm your AI-powered movie assistant. Let’s find the perfect Tamil movie for you!")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Tamil_movies_dataset.csv")
    df.columns = df.columns.str.lower()  # Standardize column names
    return df

movies_df = load_data()

# Preprocess data **(Optimized Encoding & Model Training)**
X = movies_df[['genre', 'year']]
y = movies_df['rating']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[['genre']])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['genre']))
X_encoded_df['year'] = X['year'].values

# Train model **(Trained only once to speed up recommendations)**
X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# **Precompute movie ratings once**
movies_df['predictedrating'] = regressor.predict(X_encoded_df)

# Function to recommend movies **(Optimized)**
def recommend_movies(primary_genre, min_rating, year, start_index=0, batch_size=5):
    # Use precomputed `predictedrating`
    filtered_movies = movies_df[
        (movies_df['genre'].str.contains(primary_genre, case=False, na=False)) &
        (movies_df['predictedrating'] >= min_rating) &
        (movies_df['year'] >= year)
    ].sort_values(by='predictedrating', ascending=False)

    if filtered_movies.empty:
        return pd.DataFrame()  # Return empty if no match

    return filtered_movies.iloc[start_index: start_index + batch_size][['moviename', 'genre', 'predictedrating', 'year']]

# **Session State Initialization**
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

# **Display Chat History**
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state["messages"]:
    role_class = "user-message" if message["role"] == "user" else "bot-message"
    st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# **Chat Input**
user_input = st.chat_input("💬 Type your message...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": f"✨ {user_input}"})
    
    # Default response to avoid errors
    response = "🤖 I'm not sure what you mean. Please try again!"

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
            batch_size = 5
            new_recommendations = recommend_movies(
                st.session_state["primary_genre"], st.session_state["min_rating"], st.session_state["year"],
                start_index=index, batch_size=batch_size
            )

            if new_recommendations.empty:
                response = "🤖 ❌ No more movies found! Type 'restart' to start over."
            else:
                response = "🤖 🎥 **Here are more movies for you:**\n\n"
                for _, row in new_recommendations.iterrows():
                    response += (
                        f"🎬 **{row['moviename']}**\n"
                        f"🎭 Genre: {row['genre']}\n"
                        f"⭐ Rating: {row['predictedrating']:.1f}\n"
                        f"📅 Year: {row['year']}\n\n"
                    )

                st.session_state["movie_index"] += batch_size

                if st.session_state["movie_index"] >= len(st.session_state["recommendations"]):
                    response += "✨ No more movies left! Type 'restart' to start over."
                else:
                    response += "✨ Type 'more' to see more movies or 'exit' to stop."

    # Append bot response
    st.session_state["messages"].append({"role": "assistant", "content": response})
