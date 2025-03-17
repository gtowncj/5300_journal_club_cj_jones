import streamlit as st
import joblib
import pandas as pd
import numpy as np
import random
import requests
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Set a flag to control debug output
debug = True

# Load the pretrained logistic regression model and scaler.
model = joblib.load('logistic_chess_model.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming the scaler was used during training.

st.title("Chess Outcome Predictor by Elo Group (Debug Mode ON)")

# Define the 13 Elo groups as tuples (min, max)
elo_groups = [
    (0, 200), (201, 400), (401, 600), (601, 800), (801, 1000),
    (1001, 1200), (1201, 1400), (1401, 1600), (1601, 1800),
    (1801, 2000), (2001, 2200), (2201, 2400), (2401, 9999)  # final group: 2400+
]

# ------------------------------
# Existing functionality: Fetch Player Game Details
# ------------------------------

st.header("Fetch Player Game Details")

# Input fields for username, year, and month (for individual user testing)
username_input = st.text_input("Enter username:", value="hikaru", key="details_username")
year_input = st.number_input("Year (YYYY):", 2025, 2030, value=2025, key="details_year")
month_input = st.number_input("Month (1-12):", 1, 12, value=3, key="details_month")

def get_player_games(username, year, month):
    url = f"https://api.chess.com/pub/player/{username.lower()}/games/{year}/{month:02d}"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyStreamlitApp/1.0)'}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        st.write(f"Failed to retrieve games for {username} for {year}/{month:02d}. Status code: {resp.status_code}")
        return pd.DataFrame()
    data = resp.json()
    games = data.get("games", [])
    if not games:
        # st.write("No games found for this period.")
        return pd.DataFrame()
    
    rows = []
    for g in games:
        white_result = g.get("white", {}).get("result")
        black_result = g.get("black", {}).get("result")
        # Here we also include the PGN so we can compute number of moves.
        row = {
            "game_url": g.get("url"),
            "white_username": g.get("white", {}).get("username"),
            "black_username": g.get("black", {}).get("username"),
            "white_rating": g.get("white", {}).get("rating"),
            "black_rating": g.get("black", {}).get("rating"),
            "white_result": white_result,
            "black_result": black_result,
            "time_class": g.get("time_class"),
            "pgn": g.get("pgn", ""),  # Include PGN to count moves later.
            "white_win": 1 if white_result == "win" else 0,
            "black_win": 1 if black_result == "win" else 0
        }
        rows.append(row)
    return pd.DataFrame(rows)

if st.button("Fetch Player Games"):
    df_player_games = get_player_games(username_input, int(year_input), int(month_input))
    if df_player_games.empty:
        st.write("No games to display. Check your input or try another month.")
    else:
        st.write("Player Game Details:")
        st.dataframe(df_player_games)

# ------------------------------
# Function: Find first player in Elo range using blitz stats
# ------------------------------

def find_first_player_in_elo_range(players_list, min_elo=500, max_elo=1500, game_type="chess_blitz"):
    for username in players_list:
        if username.lower() == "-anonymous-":
            continue
        stats_url = f"https://api.chess.com/pub/player/{username}/stats"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyStreamlitApp/1.0)'}
        stats_resp = requests.get(stats_url, headers=headers)
        if stats_resp.status_code != 200:
            if debug:
                print(f"Skipping {username}: unable to fetch stats from {stats_url} (status code {stats_resp.status_code}).")
            continue
        stats = stats_resp.json()
        if game_type in stats and "last" in stats[game_type]:
            rating = stats[game_type]["last"].get("rating")
            if rating is not None:
                if debug:
                    print(f"{username} has a {game_type} rating of {rating}.")
                if min_elo <= rating <= max_elo:
                    if debug:
                        print(f"Found match: {username} with rating {rating} falls in range [{min_elo}, {max_elo}].")
                        st.write(f"Found match: {username} with rating {rating} falls in range [{min_elo}, {max_elo}].")
                    return username, rating
    return None, None

# ------------------------------
# New functionality: Find 13 Players for Each Elo Group
# ------------------------------

st.header("Find US Players by Elo Group")

def get_country_players(country_code="US"):
    url = f"https://api.chess.com/pub/country/{country_code.upper()}/players"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyStreamlitApp/1.0)'}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        st.write(f"Failed to retrieve players for {country_code}. Status code: {resp.status_code}")
        return []
    data = resp.json()
    players = data.get("players", [])
    return players

def find_players_for_elo_groups(country_code="US", elo_groups=elo_groups, game_type="chess_blitz"):
    country_players = get_country_players(country_code)
    results = {}
    for group in elo_groups:
        min_elo, max_elo = group
        if max_elo <= 2000:
            player, rating = find_first_player_in_elo_range(country_players, min_elo, max_elo, game_type)
        else:
            # For higher Elo ranges, use titled players (e.g., GMs).
            titled_url = "https://api.chess.com/pub/titled/GM"
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyStreamlitApp/1.0)'}
            resp = requests.get(titled_url, headers=headers)
            if resp.status_code != 200:
                st.write("Failed to retrieve titled players.")
                player, rating = None, None
            else:
                titled_players = resp.json().get("players", [])
                player, rating = find_first_player_in_elo_range(titled_players, min_elo, max_elo, game_type)
        results[f"{min_elo}-{max_elo}"] = {"player": player, "rating": rating}
    return results

if st.button("Find US Players by Elo Group"):
    group_results = find_players_for_elo_groups(country_code="US", elo_groups=elo_groups, game_type="chess_blitz")
    st.write("Found US players for each Elo group (player and blitz rating):")
    st.write(group_results)

# ------------------------------
# New functionality: For each of the 13 found players, fetch at least 10 recent games.
# ------------------------------

st.header("Fetch at Least 10 Recent Games for Each Found Player")

def get_minimum_games_for_player(username, min_games=10):
    """
    Starting from the current month, go backward until at least min_games have been collected.
    """
    current_date = datetime.now()
    year = current_date.year
    month = current_date.month
    combined_df = pd.DataFrame()
    months_checked = 0
    while len(combined_df) < min_games and months_checked < 24:
        df = get_player_games(username, year, month)
        if not df.empty:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        months_checked += 1
        if month == 1:
            month = 12
            year -= 1
        else:
            month -= 1
    return combined_df

if st.button("Fetch Games for Players"):
    # Re-run the players search to get the latest results.
    group_results = find_players_for_elo_groups(country_code="US", elo_groups=elo_groups, game_type="chess_blitz")
    all_games_df = pd.DataFrame()
    for elo_range, info in group_results.items():
        player = info.get("player")
        rating = info.get("rating")
        if player is not None:
            st.write(f"Fetching games for {player} (Elo group {elo_range}, blitz rating {rating})")
            df_games = get_minimum_games_for_player(player, min_games=10)
            if not df_games.empty:
                df_games["elo_group"] = elo_range
                df_games["player"] = player
                all_games_df = pd.concat([all_games_df, df_games], ignore_index=True)
            else:
                st.write(f"No games found for {player}.")
        else:
            st.write(f"No player found for Elo group {elo_range}.")
    if not all_games_df.empty:
        st.write("Combined Game Data for All Found Players:")
        st.dataframe(all_games_df)
    else:
        st.write("No game data collected for any player.")

# ------------------------------
# New functionality: Predict Outcomes with the Trained Logistic Regression Model
# ------------------------------

st.header("Predict Game Outcomes for Found Players")

if st.button("Predict Games for Found Players"):

    group_results = find_players_for_elo_groups(country_code="US", elo_groups=elo_groups, game_type="chess_blitz")
    all_games_df = pd.DataFrame()
    for elo_range, info in group_results.items():
        player = info.get("player")
        rating = info.get("rating")
        if player is not None:
            st.write(f"Fetching games for {player} (Elo group {elo_range}, blitz rating {rating})")
            df_games = get_minimum_games_for_player(player, min_games=10)
            if not df_games.empty:
                df_games["elo_group"] = elo_range
                df_games["player"] = player
                all_games_df = pd.concat([all_games_df, df_games], ignore_index=True)
            else:
                st.write(f"No games found for {player}.")
        else:
            st.write(f"No player found for Elo group {elo_range}.")
    if not all_games_df.empty:
        st.write("Combined Game Data for All Found Players:")
        st.dataframe(all_games_df)
    else:
        st.write("No game data collected for any player.")

    # Use the combined DataFrame collected earlier.
    df = all_games_df.copy()
    # ------------------------------------------------------------
    # Basic Cleaning & Filtering
    # ------------------------------------------------------------
    df.drop_duplicates(inplace=True)
    # Create a 'winner' column based on the API results.
    df['winner'] = df.apply(lambda row: 'white' if row['white_result'] == 'win' 
                                else ('black' if row['black_result'] == 'win' else 'draw'), axis=1)
    # Drop rows missing critical columns
    df.dropna(subset=['white_rating', 'black_rating', 'winner', 'pgn'], inplace=True)
    # Exclude draws
    df = df[df['winner'].isin(['white', 'black'])]
    
    # ------------------------------------------------------------
    # Construct Features
    # ------------------------------------------------------------
    df['rating_diff'] = df['white_rating'] - df['black_rating']
    # Compute number of moves from PGN text (adjust the splitting as needed)
    df['num_moves'] = df['pgn'].apply(lambda x: len(x.split()) if pd.notna(x) and x.strip() != "" else np.nan)
    df.dropna(subset=['num_moves'], inplace=True)
    # Create binary outcome: 1 if white wins, 0 if black wins
    df['white_win'] = df['winner'].apply(lambda x: 1 if x == 'white' else 0)
    
    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    features = ['rating_diff', 'num_moves']
    X_new = df[features]
    X_new_scaled = scaler.transform(X_new)
    y_pred = model.predict(X_new_scaled)
    df['predicted_winner'] = ['white' if pred == 1 else 'black' for pred in y_pred]
    
    accuracy = (df['winner'] == df['predicted_winner']).mean()
    st.write(f"Prediction accuracy on these {len(df)} games: {accuracy:.2%}")
    st.dataframe(df[['game_url', 'player', 'elo_group', 'white_username', 'black_username',
                        'white_rating', 'black_rating', 'winner', 'predicted_winner']])
    
    # ROC Curve & AUC
    y_proba = model.predict_proba(X_new_scaled)[:, 1]
    auc_score = roc_auc_score(df['white_win'], y_proba)
    st.write("AUC:", auc_score)
    fpr, tpr, thresholds = roc_curve(df['white_win'], y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
