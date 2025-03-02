import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# 🎬 Page Config
st.set_page_config(page_title="🎥 Tamil Movie Bot", page_icon="🎬", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        /* Main chat container */
        .chat-container { max-width: 600px; margin: auto; padding: 15px; }
        
        /* User messages (right-aligned, green bubble) */
        .user-message { background-color: #DCF8C6; text-align: right; margin-left: auto; 
                        padding: 12px 15px; border-radius: 15px; font-weight: bold; 
                        box-shadow: 0px 2px 5px rgba(0,0,0,0.1); width: fit-content; max-width: 80%; }

        /* Bot messages (left-aligned, white bubble) */
        .bot-message { background-color: #FFFFFF; text-align: left; margin-right: auto; 
                       padding: 12px 15px; border-radius: 15px; font-weight: bold; 
                       box-shadow: 0px 2px 5px rgba(0,0,0,0.1); width: fit-content; max-width: 80%; }

        /* Message spacing */
        .message-row { display: flex; align-items: center; margin-bottom: 10px; }

        /* Custom input box */
        .chat-input { border-radius: 25px; padding: 12px; background: #f3f3f3; width: 100%; 
                      border: 1px solid #ccc; outline: none; font-size: 16px; }

        /* Movie Recommendation Table */
        .movie-table { width: 100%; text-align: left; border-collapse: collapse; margin-top: 15px; }
        .movie-table th, .movie-table td { padding: 12px; border-bottom: 1px solid #ddd; }
        .movie-table th { background-color: #f3f3f3; font-weight: bold; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Title and Greeting
st.title("🤖 Tamil Movie Recommendation Bot")
st.write("👋 **Hello!** I'm your AI-powered movie assistant. Let’s find the perfect Tamil movie for you!")
st.write("Enter the Genre 🎥 to Recommend!!!")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Tamil_movies_dataset.csv")

movies_df = load_data()
movies_df.columns = movies_df.columns.str.lower()  # Ensure column names are case-insensitive

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

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state["messages"]:
    role_class = "user-message" if message["role"] == "user" else "bot-message"
    st.markdown(f'<div class="message-row"><div class="{role_class}">{message["content"]}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Custom chat input
user_input = st.chat_input("💬 Type your message...")

if user_input:
    # Append user message (right side)
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.markdown(f'<div class="message-row"><div class="user-message">{user_input}</div></div>', unsafe_allow_html=True)

    # Process chatbot response
    if st.session_state["step"] == 1:
        st.session_state["primary_genre"] = user_input.lower()
        st.session_state["step"] = 2
        response = "🎭 Got it! What minimum rating do you prefer? (0-10)"
    
    elif st.session_state["step"] == 2:
        try:
            rating = float(user_input)
            if 0 <= rating <= 10:
                st.session_state["min_rating"] = rating
                st.session_state["step"] = 3
                response = "📅 From which year should I suggest movies?"
            else:
                response = "❌ Please enter a rating between 0 and 10."
        except ValueError:
            response = "❌ Please enter a valid number."
    
    elif st.session_state["step"] == 3:
        try:
            year = int(user_input)
            st.session_state["year"] = year
            st.session_state["step"] = 4

            # Get recommendations
            recommendations = recommend_movies(st.session_state["primary_genre"], st.session_state["min_rating"], st.session_state["year"])

            if not recommendations.empty:
                response = "🎥 **Here are your recommended movies:**"
                
                # Movie table display
                st.markdown("<table class='movie-table'><tr><th>Movie Name</th><th>Genre</th><th>Rating</th><th>Year</th></tr>", unsafe_allow_html=True)
                for _, row in recommendations.iterrows():
                    st.markdown(f"<tr><td>{row['moviename']}</td><td>{row['genre']}</td><td>⭐ {row['predictedrating']:.1f}</td><td>{row['year']}</td></tr>", unsafe_allow_html=True)
                st.markdown("</table>", unsafe_allow_html=True)

                response += "\n✨ Type 'restart' to search again!"
            else:
                response = "❌ No movies found! Type 'restart' to try again."

        except ValueError:
            response = "❌ Please enter a valid year."

    elif user_input.lower() == "restart":
        st.session_state["step"] = 1
        response = "🔄 Restarting... 👋 Hi again! What genre of movie are you looking for?"

    # Append bot response (left side)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.markdown(f'<div class="message-row"><div class="bot-message">{response}</div></div>', unsafe_allow_html=True)
