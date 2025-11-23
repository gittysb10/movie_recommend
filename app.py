import streamlit as st
import pickle

movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.npz','rb'))
movies_list = movies['title'].values


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []

    for i in movie_list:
        movie_id = i[0]
        #fetch poster from tmdb

        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies


st.title("Movie Recommendation System")

selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    for i in recommendations:
        st.write(i)


