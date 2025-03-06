import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# Set page config as the first Streamlit command
st.set_page_config(page_title="Restaurant Analysis & Recommendation", layout="wide")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data(file_path=r"C:\Users\mistr\Desktop\companies try\Cognifyz_Technologies\data\Dataset .csv"):
    try:
        df = pd.read_csv(file_path)
        # Country code mapping
        country_mapping = {
            162: "Philippines", 30: "Brazil", 216: "Qatar", 214: "Indonesia",
            184: "Sri Lanka", 189: "Turkey", 191: "UAE", 1: "India",
            14: "Australia", 37: "Canada", 94: "Malaysia", 148: "New Zealand",
            152: "South Africa", 215: "UK", 166: "USA"
        }
        df['Country'] = df['Country Code'].map(country_mapping)
        # Clean data
        df['Cuisines'] = df['Cuisines'].str.strip().str.title().fillna("Unknown")
        df['City'] = df['City'].str.strip().str.title().fillna("Unknown")
        # Combined features for recommendation
        df['features'] = df['Cuisines'] + ' ' + df['City'] + ' Price' + df['Price range'].astype(str)
        return df
    except FileNotFoundError:
        st.error("Dataset file not found at the specified path.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Precompute TF-IDF matrix
@st.cache_resource
def get_tfidf_matrix(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['features'])
    return vectorizer, tfidf_matrix

# Load data
df = load_data()
if df is None:
    st.stop()

# Precompute TF-IDF
vectorizer, tfidf_matrix = get_tfidf_matrix(df)

# --- Streamlit UI Setup ---
st.title("🍽️ Comprehensive Restaurant Analysis and Recommendation System")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = ["Home", "Restaurant Distribution", "Ratings Analysis", "Cuisine Diversity", "Restaurant Recommendation"]
choice = st.sidebar.radio("Go to", options)

# --- Pages ---
if choice == "Home":
    st.write("### Welcome to the Restaurant Analysis & Recommendation System!")
    st.write("Explore restaurant distributions, ratings, cuisine diversity, or get personalized recommendations.")

elif choice == "Restaurant Distribution":
    st.write("## Restaurant Distribution on Map")
    if "Latitude" in df.columns and "Longitude" in df.columns:
        # Filter out invalid coordinates
        map_df = df.dropna(subset=['Latitude', 'Longitude'])
        map_df = map_df[(map_df['Latitude'].between(-90, 90)) & (map_df['Longitude'].between(-180, 180))]

        if not map_df.empty:
            # Center the map on the mean of valid coordinates
            map_center = [map_df['Latitude'].mean(), map_df['Longitude'].mean()]
            restaurant_map = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")

            # Add marker clustering
            marker_cluster = MarkerCluster().add_to(restaurant_map)

            # Add markers to the cluster
            for _, row in map_df.iterrows():
                popup_info = f"<b>{row['Restaurant Name']}</b><br>City: {row['City']}<br>Rating: {row['Aggregate rating']}<br>Cuisines: {row['Cuisines']}<br>Price: {row['Price range']}"
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=popup_info,
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(marker_cluster)

            # Render the map
            folium_static(restaurant_map)
        else:
            st.error("No valid Latitude/Longitude data available to display.")
    else:
        st.error("Latitude and Longitude columns are missing.")

elif choice == "Ratings Analysis":
    st.write("## Average Ratings by City")
    country_choice = st.selectbox("Select Country", ["All Countries"] + sorted(df['Country'].dropna().unique()))

    if country_choice == "All Countries":
        avg_ratings = df.groupby("City")['Aggregate rating'].mean().sort_values(ascending=False)
    else:
        avg_ratings = df[df['Country'] == country_choice].groupby("City")['Aggregate rating'].mean().sort_values(ascending=False)

    fig = px.bar(avg_ratings.reset_index(), x="City", y="Aggregate rating", 
                 title=f"Average Restaurant Ratings in {country_choice}",
                 color="Aggregate rating", color_continuous_scale="Viridis")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Cuisine Diversity":
    st.write("## Cuisine Diversity by City")
    country_choice = st.selectbox("Select Country", ["All Countries"] + sorted(df['Country'].dropna().unique()))

    if country_choice == "All Countries":
        cuisine_diversity = df.groupby("City")['Cuisines'].nunique().sort_values(ascending=False)
    else:
        cuisine_diversity = df[df['Country'] == country_choice].groupby("City")['Cuisines'].nunique().sort_values(ascending=False)

    fig = px.bar(cuisine_diversity.reset_index(), x="City", y="Cuisines", 
                 title=f"Cuisine Diversity in {country_choice}",
                 color="Cuisines", color_continuous_scale="Teal")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

elif choice == "Restaurant Recommendation":
    st.write("## Find a Restaurant Recommendation")
    cuisine_choice = st.selectbox("Select Cuisine Type", sorted(df['Cuisines'].unique()))
    available_localities = df[df['Cuisines'].str.contains(cuisine_choice, case=False, na=False)]['City'].unique()
    
    if len(available_localities) > 0:
        locality_choice = st.selectbox("Select Locality", sorted(available_localities))
        available_price_ranges = df[(df['Cuisines'].str.contains(cuisine_choice, case=False, na=False)) & 
                                    (df['City'] == locality_choice)]['Price range'].unique()
        price_choice = st.selectbox("Select Price Range", sorted(available_price_ranges))

        # Recommendation logic with fuzzy matching
        df['Cuisine Match'] = df['Cuisines'].apply(lambda x: fuzz.partial_ratio(x, cuisine_choice))
        user_input = f"{cuisine_choice} {locality_choice} Price{price_choice}"
        user_vector = vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
        df['Similarity Score'] = similarity_scores

        # Filter and rank recommendations
        recommended = df[(df['Cuisine Match'] >= 70) & (df['City'] == locality_choice) & 
                         (df['Price range'] == price_choice)].sort_values(by='Similarity Score', ascending=False).head(10)

        if not recommended.empty:
            st.subheader("Recommended Restaurants")
            st.dataframe(recommended[['Restaurant Name', 'Cuisines', 'City', 'Price range', 'Aggregate rating']], 
                        use_container_width=True)

            # Top 10 by rating
            recommended_sorted = recommended.sort_values(by='Aggregate rating', ascending=False)
            fig = px.bar(recommended_sorted, x="Aggregate rating", y="Restaurant Name", orientation='h',
                         title="Top 10 Recommendations by Rating", color="Aggregate rating", 
                         color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No restaurants match your criteria.")
    else:
        st.warning(f"No restaurants found for '{cuisine_choice}'.")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.text("Developed by Meet Mistry")
st.sidebar.markdown("---")