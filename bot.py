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
        /* Movie card styling */
        .movie-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 12px;
            border-left: 4px solid #ff5722;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .movie-title {
            font-weight: bold;
            margin-bottom: 5px;
            color: #333333;
        }
        .movie-info {
            margin-left: 10px;
            color: #555555;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Greeting
st.title("🤖 Tamil Movie Recommendation Bot")
st.write("👋 **Hello!** I'm your AI-powered movie assistant. Let's find the perfect Tamil movie for you!")
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

# Display chat history (Styled as WhatsApp chat)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state["messages"]:
    role_class = "user-message" if message["role"] == "user" else "bot-message"
    # Removed prefix from here as we'll add it directly in the content
    st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Custom chat input
user_input = st.chat_input("💬 Type your message...")

if user_input:
    # Add user message with emoji prefix (but not in the displayed content)
    st.session_state["messages"].append({"role": "user", "content": f"👤 {user_input}"})

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
            st.session_state["step"] = 4

            # Get recommendations
            recommendations = recommend_movies(st.session_state["primary_genre"], st.session_state["min_rating"], st.session_state["year"])

            if not recommendations.empty:
                response = "🤖 🎥 **Here are your recommended movies:**\n"

                # Limit to top 5 recommendations for better display
                max_recommendations = min(5, len(recommendations))
                
                for i in range(max_recommendations):
                    row = recommendations.iloc[i]
                    response += f"""
                    🎬 **{row['moviename']}**  
                    🎭 **Genre:** {row['genre']}  
                    ⭐ **Rating:** {row['predictedrating']:.1f}  
                    📅 **Year:** {row['year']}  
                    \n---
                    """
                
                if len(recommendations) > max_recommendations:
                    response += f"*...and {len(recommendations) - max_recommendations} more movies*\n\n"
                
                response += "✨ Type **'restart'** to search again!"
            else:
                response = "🤖 ❌ No movies found! Type 'restart' to try again."

        except ValueError:
            response = "🤖 ❌ Please enter a valid year."

    elif user_input.lower() == "restart":
        st.session_state["step"] = 1
        response = "🤖 🔄 Restarting... 👋 Hi again! What genre of movie are you looking for? 🎭"
    else:
        # Handle any other input when in recommendation state
        if st.session_state["step"] == 4:
            response = "🤖 Type 'restart' to search for different movies!"
        else:
            response = "🤖 I'm not sure how to respond to that. Let's continue with the current step."

    # Add bot response to messages - note the emoji is already in the response
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Re-render the chat with updated messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state["messages"]:
        role_class = "user-message" if message["role"] == "user" else "bot-message"
        # No prefix added here - the content already has the emoji
        st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
