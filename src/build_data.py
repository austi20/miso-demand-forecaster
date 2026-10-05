"""Rebuild both raw parquet files: python -m src.build_data"""

from src import pull_eia, pull_weather

if __name__ == "__main__":
    pull_eia.main()
    pull_weather.main()
