import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------
# LOAD MOVIE DATA
# -----------------------------------------

movies = pd.read_csv("movies.csv")


# -----------------------------------------
# COMBINE FEATURES
# -----------------------------------------

movies["features"] = (
    movies["genre"] + " " + movies["description"]
)


# -----------------------------------------
# CONVERT TEXT INTO NUMBERS
# -----------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english"
)

feature_matrix = vectorizer.fit_transform(
    movies["features"]
)


# -----------------------------------------
# CALCULATE SIMILARITY
# -----------------------------------------

similarity = cosine_similarity(feature_matrix)


# -----------------------------------------
# RECOMMENDATION FUNCTION
# -----------------------------------------

def recommend_movies(movie_name, number_of_recommendations=5):

    movie_name = movie_name.lower()

    # Find the movie
    movie_index = -1

    for index, title in enumerate(movies["title"]):

        if title.lower() == movie_name:
            movie_index = index
            break

    # Movie not found
    if movie_index == -1:

        print("\n❌ Movie not found.")

        print("\nAvailable movies:")

        for title in movies["title"]:
            print("-", title)

        return

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity[movie_index])
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\n🎬 Recommended Movies")
    print("--------------------------------")

    count = 0

    for index, score in similarity_scores:

        # Skip the selected movie
        if index == movie_index:
            continue

        print(
            f"{count + 1}. "
            f"{movies.iloc[index]['title']} "
            f"(Similarity: {score:.2f})"
        )

        count += 1

        if count == number_of_recommendations:
            break


# -----------------------------------------
# MAIN PROGRAM
# -----------------------------------------

if __name__ == "__main__":

    print("================================")
    print("     MOVIE RECOMMENDATION SYSTEM")
    print("================================")

    print("\nAvailable movies:")

    for title in movies["title"]:
        print("-", title)

    movie = input(
        "\nEnter a movie you like: "
    )

    recommend_movies(movie)