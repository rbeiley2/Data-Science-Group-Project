# Milestone 1: Data preprocessing for the NBA Games dataset
#
# What this script does:
#   1. Loads all 5 CSV files (games, games_details, players, ranking, teams).
#   2. Checks each one for missing values and duplicates.
#   3. Handles the missing data. For each column we first ask WHY the value is missing,
#      then decide whether to fill it, flag it, or drop it (instead of deleting rows by default).
#   4. Joins the tables to check that the IDs line up.
#   5. Saves the cleaned files to data/clean/.
#
# How to run: put the unzipped NBA_dataset folder in data/raw/, then press Run.

from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw" / "NBA_dataset"
CLEAN = PROJECT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

if not (RAW / "games.csv").exists():
    raise FileNotFoundError(f"Put the unzipped NBA_dataset folder here: {RAW}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def missing_summary(df, name):
    """Print the shape of a dataframe and how many values are missing in each column."""
    print(f"\n{name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  no missing values")
    else:
        print(pd.DataFrame({"missing": missing, "percent": (missing / len(df) * 100).round(1)}))


# ----------------------------------------------------------------------------
# 1. Load the data
# ----------------------------------------------------------------------------
section("1. LOAD ALL 5 FILES")
teams = pd.read_csv(RAW / "teams.csv")
players = pd.read_csv(RAW / "players.csv")
ranking = pd.read_csv(RAW / "ranking.csv")
games = pd.read_csv(RAW / "games.csv")
details = pd.read_csv(RAW / "games_details.csv", low_memory=False)

for name, df in [("teams", teams), ("players", players), ("ranking", ranking),
                 ("games", games), ("games_details", details)]:
    missing_summary(df, name)
    # Which columns have at least one missing value? (axis=0 checks each column)
    print("  columns with missing values:", list(df.columns[df.isnull().any(axis=0)]))


# ----------------------------------------------------------------------------
# 2. teams.csv
# ----------------------------------------------------------------------------
section("2. TEAMS")
print(teams[["ABBREVIATION", "ARENACAPACITY"]].sort_values("ARENACAPACITY").head(6))

# Orlando's arena capacity is 0, which is impossible. A 0 here really means "unknown",
# so we turn it into a missing value first.
teams["ARENACAPACITY"] = teams["ARENACAPACITY"].replace(0, np.nan)

# 5 teams now have an unknown capacity. Arena size is just background info about a team
# (it is not something we will predict with), so filling with the median is a safe, simple choice.
# We keep a flag so we always know which values were filled in.
teams["ARENACAPACITY_FILLED"] = teams["ARENACAPACITY"].isnull().astype(int)
teams["ARENACAPACITY"] = teams["ARENACAPACITY"].fillna(teams["ARENACAPACITY"].median())
print("filled arena capacities:", teams["ARENACAPACITY_FILLED"].sum())


# ----------------------------------------------------------------------------
# 3. players.csv
# ----------------------------------------------------------------------------
section("3. PLAYERS")
# No missing values. Check for duplicates and which seasons are covered.
print("duplicate rows:", players.duplicated().sum())
print("seasons covered:", players["SEASON"].min(), "to", players["SEASON"].max())
# This file only covers 2009-2019, but the games go from 2003 to 2025.
# So for player work across all seasons, we will use games_details instead (it has every player in every game).


# ----------------------------------------------------------------------------
# 4. games.csv
# ----------------------------------------------------------------------------
section("4. GAMES")
games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"])

# The first digit of GAME_ID tells us the type of game.
game_types = {1: "Preseason", 2: "Regular Season", 4: "Playoffs", 5: "Play-In", 6: "NBA Cup Final"}
games["GAME_TYPE"] = (games["GAME_ID"] // 10_000_000).map(game_types)
print(games["GAME_TYPE"].value_counts())

# Duplicates: 29 games from Dec 2020 were recorded twice. Keep one copy of each.
print("\nduplicate GAME_IDs:", games["GAME_ID"].duplicated().sum())
games = games.drop_duplicates(subset="GAME_ID")

# TEAM_ID_home and TEAM_ID_away are exact copies of HOME_TEAM_ID and VISITOR_TEAM_ID, so drop them.
print("TEAM_ID_home is a copy of HOME_TEAM_ID:", (games["TEAM_ID_home"] == games["HOME_TEAM_ID"]).all())
games = games.drop(columns=["TEAM_ID_home", "TEAM_ID_away"])

# Missing data: 99 games have ALL their stats missing (points, shooting, rebounds...).
no_stats = games[games["PTS_home"].isnull()]
print("\ngames with no stats:", len(no_stats))
print("  game type:", no_stats["GAME_TYPE"].unique().tolist())
print("  dates:", no_stats["GAME_DATE_EST"].min().date(), "to", no_stats["GAME_DATE_EST"].max().date())
print("  HOME_TEAM_WINS in these rows:", no_stats["HOME_TEAM_WINS"].unique().tolist())

# All 99 are preseason games from Oct 2003, and all of them say the home team lost (0).
# Home teams win about 58% of games, so that 0 is a placeholder, not a real result.
# We can't fill in the scores (that would invent results), and they aren't in games_details either.
# Decision: keep the rows, flag them, and mark the result as unknown.
# We won't use preseason games for modeling, so nothing useful is lost.
games["STATS_MISSING"] = games["PTS_home"].isnull().astype(int)
games["HOME_TEAM_WINS"] = games["HOME_TEAM_WINS"].where(games["STATS_MISSING"] == 0).astype("Int64")

# Sanity check: does HOME_TEAM_WINS match the score in every other game?
scored = games[games["STATS_MISSING"] == 0]
print("label matches the score:", ((scored["PTS_home"] > scored["PTS_away"]).astype(int) == scored["HOME_TEAM_WINS"]).all())
print(games[["PTS_home", "PTS_away", "FG_PCT_home", "REB_home"]].describe().loc[["min", "mean", "max"]])


# ----------------------------------------------------------------------------
# 5. games_details.csv (one row per player per game)
# ----------------------------------------------------------------------------
section("5. GAMES_DETAILS")

# Duplicates: the same Dec 2020 games, recorded twice.
print("duplicate player rows:", details.duplicated(subset=["GAME_ID", "PLAYER_ID"]).sum())
details = details.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])

# Missing data #1: about 125,000 rows have ALL stats missing.
# The COMMENT column explains why: the player did not play (DNP), did not dress (DND),
# or was not with the team (NWT).
details["COMMENT"] = details["COMMENT"].str.strip()
print("\nrows with no stats:", details["MIN"].isnull().sum())
print(details.loc[details["MIN"].isnull(), "COMMENT"].value_counts().head(5))

# These stats are missing because they CAN'T exist (the player never got on the court).
# We do NOT fill them with 0: that would make players' per-game averages look worse than they are.
# Instead we keep the rows and add a PLAYED flag and a simple reason column.
details["PLAYED"] = details["MIN"].notnull().astype(int)


def dnp_reason(comment):
    if pd.isnull(comment):
        return "Unknown"
    comment = comment.lower()
    if "coach" in comment:
        return "Coach's Decision"
    if "rest" in comment:
        return "Rest"
    if "not with team" in comment or comment.startswith("nwt"):
        return "Not With Team"
    return "Injury/Illness"


details["DNP_REASON"] = details["COMMENT"].apply(dnp_reason).where(details["PLAYED"] == 0)
print("\nwhy players didn't play:")
print(details["DNP_REASON"].value_counts())

# MIN is text in different formats ("34:12", "34.000000:12", "34"). Convert to decimal minutes.
def to_minutes(text):
    if pd.isnull(text):
        return np.nan
    if ":" in text:
        minutes, seconds = text.split(":")
        return float(minutes) + float(seconds) / 60
    return float(text)


details["MIN"] = details["MIN"].apply(to_minutes)

# A few minute values are impossible: negative, or over 68 (a 4-overtime game is 68 minutes).
# These are typos, so we set them to missing (the player's other stats are fine).
bad_minutes = (details["MIN"] < 0) | (details["MIN"] > 68)
print("\nimpossible minute values set to missing:", bad_minutes.sum())
details.loc[bad_minutes, "MIN"] = np.nan

# Missing data #2 (hidden): when a player takes 0 shots, the shooting % is stored as 0.
# But 0 made out of 0 attempts is undefined, not 0%. Turn these into missing values.
for pct, attempts in [("FG_PCT", "FGA"), ("FG3_PCT", "FG3A"), ("FT_PCT", "FTA")]:
    zero_attempts = (details["PLAYED"] == 1) & (details[attempts] == 0)
    print(f"{pct}: {zero_attempts.sum():,} rows with 0 attempts changed from 0 to missing")
    details.loc[zero_attempts, pct] = np.nan

# START_POSITION is empty for bench players (every team has exactly 5 starters per game).
# So empty doesn't mean unknown, it means "Bench".
details["START_POSITION"] = details["START_POSITION"].fillna("Bench")

# NICKNAME is empty for every season before 2020, and we already have PLAYER_NAME. Drop it.
details = details.drop(columns=["NICKNAME"])

# PLUS_MINUS is missing for some players who did play. All of these are preseason games
# from 2003-2013, where the NBA didn't track it. We leave it missing and won't use preseason games.
played = details[details["PLAYED"] == 1]
pm_missing = played[played["PLUS_MINUS"].isnull()]
print("\nPLUS_MINUS missing for players who played:", len(pm_missing))
print("  game types:", (pm_missing["GAME_ID"] // 10_000_000).map(game_types).unique().tolist())


# ----------------------------------------------------------------------------
# 6. ranking.csv (standings for every team on every day)
# ----------------------------------------------------------------------------
section("6. RANKING")
ranking["STANDINGSDATE"] = pd.to_datetime(ranking["STANDINGSDATE"])
ranking["SEASON"] = ranking["SEASON_ID"] % 10000  # SEASON_ID 22019 -> season 2019

# RETURNTOPLAY is 98% missing. It only exists for the 2019-20 COVID "bubble" season,
# so it's useless for every other season. Drop it.
print("seasons that have RETURNTOPLAY:", ranking.loc[ranking["RETURNTOPLAY"].notnull(), "SEASON"].unique().tolist())
ranking = ranking.drop(columns=["RETURNTOPLAY"])

# Duplicates (Dec 2020 again).
print("duplicate team-date rows:", ranking.duplicated(subset=["TEAM_ID", "STANDINGSDATE"]).sum())
ranking = ranking.drop_duplicates(subset=["TEAM_ID", "STANDINGSDATE"])

# Hidden missing data: on opening day every team is 0-0 and W_PCT is stored as 0.
# A team that hasn't played has no win %, so set it to missing and add a flag.
# (For the prediction model later, we can fill these with last season's win %.)
print("rows with 0 games played:", (ranking["G"] == 0).sum())
ranking["NO_GAMES_YET"] = (ranking["G"] == 0).astype(int)
ranking.loc[ranking["G"] == 0, "W_PCT"] = np.nan

# HOME_RECORD / ROAD_RECORD are text like "20-5". Split them into wins and losses.
ranking[["HOME_W", "HOME_L"]] = ranking["HOME_RECORD"].str.split("-", expand=True).astype(int)
ranking[["ROAD_W", "ROAD_L"]] = ranking["ROAD_RECORD"].str.split("-", expand=True).astype(int)
print(ranking[["TEAM", "STANDINGSDATE", "W", "L", "W_PCT", "HOME_W", "HOME_L"]].head())


# ----------------------------------------------------------------------------
# 7. Joins: do the IDs match across files?
# ----------------------------------------------------------------------------
section("7. JOINS")
print("every game's team is in teams.csv:", games["HOME_TEAM_ID"].isin(teams["TEAM_ID"]).all())
print("every player row's game is in games.csv:", details["GAME_ID"].isin(games["GAME_ID"]).all())

# Add team names to each game (merge games with teams).
games = games.merge(teams[["TEAM_ID", "ABBREVIATION"]].rename(columns={"TEAM_ID": "HOME_TEAM_ID", "ABBREVIATION": "HOME_TEAM"}),
                    on="HOME_TEAM_ID", how="left")
games = games.merge(teams[["TEAM_ID", "ABBREVIATION"]].rename(columns={"TEAM_ID": "VISITOR_TEAM_ID", "ABBREVIATION": "AWAY_TEAM"}),
                    on="VISITOR_TEAM_ID", how="left")
print(games[["GAME_DATE_EST", "HOME_TEAM", "AWAY_TEAM", "PTS_home", "PTS_away", "HOME_TEAM_WINS"]].head())

# Example of what the joined data can do: average points per game for each team (regular season).
regular = games[games["GAME_TYPE"] == "Regular Season"]
print("\nhighest scoring home teams (regular season):")
print(regular.groupby("HOME_TEAM")["PTS_home"].agg(["mean", "count"]).round(1).nlargest(5, "mean"))


# ----------------------------------------------------------------------------
# 8. Summary and save
# ----------------------------------------------------------------------------
section("8. AFTER CLEANING")
for name, df in [("teams", teams), ("players", players), ("ranking", ranking),
                 ("games", games), ("games_details", details)]:
    missing_summary(df, name)
    df.to_csv(CLEAN / f"{name}_clean.csv", index=False)

print(f"\nSaved cleaned files to {CLEAN}")
