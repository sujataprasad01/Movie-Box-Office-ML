import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def custom_rerun():
    st.session_state.rerun = True 


def safe_literal_eval(val):
    try:
        return len(ast.literal_eval(val)) if isinstance(ast.literal_eval(val), list) else 0
    except (ValueError, SyntaxError):
        return 0


def format_sales(sales):
    if sales >= 10**7:
        return f"{sales/10**7:.2f} Crores"
    elif sales >= 10**5:
        return f"{sales/10**5:.2f} Lakh"
    elif sales >= 10**3:
        return f"{sales/10**3:.2f} K"
    else:
        return str(sales)


with open('box_office_model.pkl', 'rb') as file:
    model = pickle.load(file)

train = pd.read_csv('train.csv')
np.random.seed(42)

train['ticket_sales'] = np.random.normal(
    loc=train['budget'].mean() * 0.8,
    scale=train['budget'].std() * 0.2,
    size=len(train)
).clip(50000, 1000000).astype(int)

train['release_date'] = pd.to_datetime(train['release_date'], errors='coerce')
train['production_companies_count'] = train['production_companies'].apply(safe_literal_eval) if 'production_companies' in train.columns else 0


def generate_movie_info(movie_name):
    movie_data = train[train['title'].str.contains(movie_name, case=False, na=False)]
    if movie_data.empty:
        return "Movie not found!", None

    movie = movie_data.iloc[0]
    release_date = movie['release_date']
    release_year, release_month, release_day = (release_date.year, release_date.month, release_date.day) if pd.notnull(release_date) else ('Unknown', 'Unknown', 'Unknown')

    features = pd.DataFrame([[
        movie['budget'], movie['popularity'], movie['runtime'], release_year, release_month, release_day, movie['production_companies_count']
    ]], columns=['budget', 'popularity', 'runtime', 'release_year', 'release_month', 'release_day', 'production_companies_count'])

    predicted_revenue = model.predict(features)[0]

    return {
        "title": movie['title'],
        "director": movie.get('director', 'Unknown'),
        "genres": movie.get('genres', 'Unknown'),
        "release_date": release_date,
        "ticket_sales": movie['ticket_sales'],
        "original_revenue": movie['revenue'],
        "predicted_revenue": predicted_revenue,
    }


page = st.sidebar.radio("Movie Box Office ", ["Home", "Login", "Signup", "About Us", "Contact Us"])

if page == "Home":
    st.title("Movie Box Office")
    if 'logged_in' not in st.session_state:
        st.warning("You need to be logged in to view this page.")
    else:
        movie_name = st.text_input("Enter a Movie Name:")
        if movie_name:
            movie_info = generate_movie_info(movie_name)
            if movie_info == "Movie not found!":
                st.error("Movie not found!")
            else:
                st.write(f"**Title:** {movie_info['title']}")
                st.write(f"**Director:** {movie_info['director']}")
                st.write(f"**Genres:** {movie_info['genres']}")
                st.write(f"**Release Date:** {movie_info['release_date']}")
                st.write(f"**Ticket Sales:** {format_sales(movie_info['ticket_sales'])}")
                st.write(f"**Original Revenue:** ${movie_info['original_revenue']}")
                st.write(f"**Predicted Revenue:** ${movie_info['predicted_revenue']}")

                fig, ax = plt.subplots()
                ax.bar([movie_info["title"]], [movie_info["ticket_sales"]], color="blue")
                ax.set_title("Ticket Sales")
                ax.set_ylabel("Sales")
                st.pyplot(fig)

                fig, ax = plt.subplots()
                revenue_labels = ["Original Revenue", "Predicted Revenue"]
                revenue_values = [movie_info["original_revenue"], movie_info["predicted_revenue"]]
                ax.bar(revenue_labels, revenue_values, color=["green", "orange"])
                ax.set_title(f"Revenue Breakdown for {movie_info['title']}")
                ax.set_ylabel("Revenue")
                st.pyplot(fig)

        st.subheader("Top 20 Most Popular Movies")
        popular_movies = train[['title', 'popularity']].sort_values(by='popularity', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(popular_movies['title'], popular_movies['popularity'], color='skyblue')
        ax.set_xlabel("Popularity")
        ax.set_ylabel("Movie Title")
        ax.set_title("Top 20 Most Popular Movies")
        st.pyplot(fig)

        st.subheader("Genre Distribution")
        genres_count = train['genres'].value_counts().head(10)
        fig, ax = plt.subplots()
        ax.pie(genres_count, labels=genres_count.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Top 10 Genres Distribution")
        st.pyplot(fig)

        top_movies = train.nlargest(10, "ticket_sales")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(top_movies['title'], top_movies['ticket_sales'], color='blue')
        ax.set_xlabel('Movie Title')
        ax.set_ylabel('Ticket Sales')
        ax.set_title('Top 20 Movies by Ticket Sales')
        plt.xticks(rotation=45, ha='right')
        ax.set_yticklabels([format_sales(y) for y in ax.get_yticks()])
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(train['budget'], train['revenue'], color='purple', alpha=0.6)
        ax.set_xlabel('Budget')
        ax.set_ylabel('Revenue')
        ax.set_title('Revenue vs. Budget for Movies')
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(train['revenue'].dropna(), bins=20, color='green', edgecolor='black')
        ax.set_xlabel('Revenue')
        ax.set_ylabel('Number of Movies')
        ax.set_title('Revenue Distribution of Movies')
        st.pyplot(fig)

        train['release_date'] = pd.to_datetime(train['release_date'], errors='coerce')
        sales_by_year = train.groupby(train['release_date'].dt.year)['ticket_sales'].sum()

        fig, ax = plt.subplots(figsize=(8, 6))
        sales_by_year.plot(kind='line', color='red', ax=ax)
        ax.set_xlabel('Release Year')
        ax.set_ylabel('Ticket Sales')
        ax.set_title('Ticket Sales Over the Years')
        st.pyplot(fig)

        top_genre_popularity = train.groupby('genres')['popularity'].mean().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 6))
        top_genre_popularity.plot(kind='barh', color='orange', ax=ax)
        ax.set_xlabel('Average Popularity')
        ax.set_ylabel('Genre')
        ax.set_title('Top 10 Most Popular Genres')
        st.pyplot(fig)

elif page == "Login":
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        st.session_state['logged_in'] = True
        st.success("Successfully logged in!")
        st.rerun()

elif page == "Signup":
    st.title("Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Signup"):
        if password == confirm_password:
            st.success("Signup successful!")
        else:
            st.error("Passwords do not match!")

elif page == "About Us":
    st.write("A system to predict movie revenues based on key features like budget, popularity, etc.")

elif page == "Contact Us":
    st.write("Contact us at: contact@movieboxoffice.com")
