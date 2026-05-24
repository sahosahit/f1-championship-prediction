"""
F1 Championship Data Collection
Collects race results and championship standings for seasons 2014-2026.
Uses FastF1 API to get official F1 data.
"""

import fastf1
import pandas as pd
import numpy as np
import os

# Setup directories
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def log(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


def get_race_results(year, round_number):
    """Get race results for a specific race."""
    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load(telemetry=False, weather=False, messages=False, laps=False)
        results = session.results

        if results is None or results.empty:
            return None

        race_data = results[['DriverNumber', 'Abbreviation', 'TeamName',
                            'Position', 'Points', 'Status', 'GridPosition']].copy()
        race_data['Year'] = year
        race_data['Round'] = round_number
        race_data['EventName'] = session.event['EventName']

        return race_data
    except Exception as e:
        log(f"    Error: {e}")
        return None


def build_standings(race_results_df):
    """
    Build cumulative championship standings after each race.
    Computes: points, wins, podiums, DNFs, position, gap to leader, rates.
    """
    all_standings = []

    for year in sorted(race_results_df['Year'].unique()):
        year_results = race_results_df[race_results_df['Year'] == year]
        rounds = sorted(year_results['Round'].unique())

        cumulative = {}

        for round_num in rounds:
            round_results = year_results[year_results['Round'] == round_num]

            for _, row in round_results.iterrows():
                driver = row['Abbreviation']
                team = row['TeamName']
                points = float(row['Points']) if pd.notna(row['Points']) else 0.0
                position = int(row['Position']) if pd.notna(row['Position']) and row['Position'] > 0 else 99

                if driver not in cumulative:
                    cumulative[driver] = {
                        'points': 0, 'team': team, 'wins': 0,
                        'podiums': 0, 'dnfs': 0, 'races': 0
                    }

                cumulative[driver]['points'] += points
                cumulative[driver]['team'] = team
                cumulative[driver]['races'] += 1

                if position == 1:
                    cumulative[driver]['wins'] += 1
                if position <= 3:
                    cumulative[driver]['podiums'] += 1
                if position == 99:
                    cumulative[driver]['dnfs'] += 1

            # Record standings after this round
            for driver, stats in cumulative.items():
                all_standings.append({
                    'Year': year,
                    'Round': round_num,
                    'Driver': driver,
                    'Team': stats['team'],
                    'CumulativePoints': stats['points'],
                    'Wins': stats['wins'],
                    'Podiums': stats['podiums'],
                    'DNFs': stats['dnfs'],
                    'RacesCompleted': stats['races'],
                })

    standings_df = pd.DataFrame(all_standings)

    # Add computed features
    enriched = []
    for (year, round_num), group in standings_df.groupby(['Year', 'Round']):
        group = group.copy()
        group = group.sort_values('CumulativePoints', ascending=False).reset_index(drop=True)
        group['ChampionshipPosition'] = group.index + 1

        max_pts = group['CumulativePoints'].max()
        group['PointsGapToLeader'] = max_pts - group['CumulativePoints']

        group['PointsGapToAhead'] = group['CumulativePoints'].shift(1) - group['CumulativePoints']
        group['PointsGapToAhead'] = group['PointsGapToAhead'].fillna(0)

        group['WinRate'] = group['Wins'] / group['RacesCompleted']
        group['PodiumRate'] = group['Podiums'] / group['RacesCompleted']
        group['PointsPerRace'] = group['CumulativePoints'] / group['RacesCompleted']

        enriched.append(group)

    return pd.concat(enriched, ignore_index=True)


def get_championship_winner(standings_df, year):
    """Get the championship winner for a given year."""
    year_data = standings_df[standings_df['Year'] == year]
    final_round = year_data['Round'].max()
    final_standings = year_data[year_data['Round'] == final_round]
    winner = final_standings.sort_values('CumulativePoints', ascending=False).iloc[0]
    return winner['Driver']


def collect_all_data(start_year=2014, end_year=2026):
    """
    Main function to collect all F1 championship data.

    Args:
        start_year: First season to collect (default 2014)
        end_year: Last season to collect (default 2026)
    """
    log(f"Collecting F1 data from {start_year} to {end_year}...")
    log("=" * 60)

    all_race_results = []

    for year in range(start_year, end_year + 1):
        log(f"\nSeason {year}:")

        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['RoundNumber'] > 0]
            log(f"  {len(races)} race weekends found")

            for _, event in races.iterrows():
                round_num = event['RoundNumber']
                event_name = event['EventName']
                log(f"  R{round_num:02d}: {event_name}... ")

                results = get_race_results(year, round_num)

                if results is not None and not results.empty:
                    all_race_results.append(results)
                    log(f"       OK ({len(results)} drivers)")
                else:
                    log(f"       SKIPPED")
                    if year == end_year:
                        log(f"  -> End of available {year} data (Race {round_num-1} is latest)")
                        break

        except Exception as e:
            log(f"  Error with season {year}: {e}")
            continue

    if not all_race_results:
        log("No data collected!")
        return None, None

    # Combine all results
    race_results_df = pd.concat(all_race_results, ignore_index=True)
    log(f"\n{'='*60}")
    log(f"Total race results: {len(race_results_df)} rows")
    log(f"Seasons: {race_results_df['Year'].min()} - {race_results_df['Year'].max()}")

    total_races = race_results_df.groupby(['Year', 'Round']).ngroups
    log(f"Total races: {total_races}")

    # Save raw results
    race_results_path = os.path.join(DATA_DIR, 'race_results.csv')
    race_results_df.to_csv(race_results_path, index=False)
    log(f"\nSaved race results to: {race_results_path}")

    # Build championship standings
    log("\nBuilding championship standings...")
    standings_df = build_standings(race_results_df)

    standings_path = os.path.join(DATA_DIR, 'championship_standings.csv')
    standings_df.to_csv(standings_path, index=False)
    log(f"Saved championship standings to: {standings_path}")

    # Print champions
    log(f"\n{'='*60}")
    log("CHAMPIONSHIP WINNERS:")
    log(f"{'='*60}")
    for year in sorted(standings_df['Year'].unique()):
        year_data = standings_df[standings_df['Year'] == year]
        max_round = year_data['Round'].max()
        final = year_data[year_data['Round'] == max_round]
        winner = final.sort_values('CumulativePoints', ascending=False).iloc[0]
        log(f"  {year}: {winner['Driver']} ({winner['Team']}) - {winner['CumulativePoints']:.0f} pts after {max_round} races")

    # Print current standings for the latest season
    latest_year = standings_df['Year'].max()
    log(f"\n{'='*60}")
    log(f"{latest_year} CURRENT STANDINGS (After latest race):")
    log(f"{'='*60}")
    latest_data = standings_df[standings_df['Year'] == latest_year]
    latest_round = latest_data['Round'].max()
    current = latest_data[latest_data['Round'] == latest_round]
    current = current.sort_values('CumulativePoints', ascending=False).head(10)
    log(f"  After Round {latest_round}:")
    for _, row in current.iterrows():
        log(f"    P{int(row['ChampionshipPosition']):2d}: {row['Driver']:4s} ({row['Team']:20s}) - {row['CumulativePoints']:6.1f} pts | Wins: {int(row['Wins'])} | PPR: {row['PointsPerRace']:.1f}")

    log(f"\n{'='*60}")
    log("DATA COLLECTION COMPLETE!")
    log(f"{'='*60}")
    log(f"Standings shape: {standings_df.shape}")
    log(f"Columns: {list(standings_df.columns)}")

    return race_results_df, standings_df


if __name__ == "__main__":
    collect_all_data(start_year=2014, end_year=2026)
