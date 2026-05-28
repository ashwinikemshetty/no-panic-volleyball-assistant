"""
Unit tests for schedule query functionality.

Run with: pytest test_schedule.py -v
"""

import pytest
from datetime import datetime, date, timedelta
from schedule_loader import ScheduleLoader, resolve_date
import os
import pandas as pd


class TestDateResolution:
    """Test date string parsing and resolution."""

    def test_today(self):
        assert resolve_date("today") == datetime.now().date()

    def test_today_case_insensitive(self):
        assert resolve_date("TODAY") == datetime.now().date()
        assert resolve_date("Today") == datetime.now().date()

    def test_tomorrow(self):
        assert resolve_date("tomorrow") == datetime.now().date() + timedelta(days=1)

    def test_yesterday(self):
        assert resolve_date("yesterday") == datetime.now().date() - timedelta(days=1)

    def test_mm_dd_yyyy_format(self):
        result = resolve_date("05/27/2026")
        assert result == date(2026, 5, 27)

    def test_dd_mm_yyyy_format(self):
        result = resolve_date("27/05/2026")
        assert result == date(2026, 5, 27)

    def test_month_day_year(self):
        result = resolve_date("May 27, 2026")
        assert result == date(2026, 5, 27)

    def test_month_day(self):
        result = resolve_date("May 27")
        # Should assume current year
        assert result.month == 5 and result.day == 27

    def test_ordinal_day(self):
        result = resolve_date("27th")
        # Should resolve to nearest 27th
        assert result.day == 27

    def test_none_for_empty_string(self):
        assert resolve_date("") is None
        assert resolve_date(None) is None

    def test_none_for_invalid(self):
        assert resolve_date("invalid date string") is None


class TestScheduleLoaderBasics:
    """Test ScheduleLoader initialization and data loading."""

    @pytest.fixture
    def schedule_loader(self):
        docs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "docs"
        )
        xlsx_path = os.path.join(docs_dir, "CrossbarCourtSchedule.xlsx")
        return ScheduleLoader(xlsx_path)

    def test_loader_initializes(self, schedule_loader):
        assert schedule_loader.df is not None
        assert len(schedule_loader.df) > 0

    def test_column_normalization(self, schedule_loader):
        # Columns should be lowercase with underscores
        assert "date" in schedule_loader.df.columns
        assert "teams" in schedule_loader.df.columns
        assert "space" in schedule_loader.df.columns
        assert "start" in schedule_loader.df.columns
        assert "end" in schedule_loader.df.columns

    def test_date_parsing(self, schedule_loader):
        # All dates should be datetime objects
        assert pd.api.types.is_datetime64_any_dtype(schedule_loader.df["date"])


class TestScheduleQueries:
    """Test schedule query methods."""

    @pytest.fixture
    def schedule_loader(self):
        docs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "docs"
        )
        xlsx_path = os.path.join(docs_dir, "CrossbarCourtSchedule.xlsx")
        return ScheduleLoader(xlsx_path)

    def test_query_by_team_exact(self, schedule_loader):
        # Should find at least some 14-3 Girls sessions
        results = schedule_loader.query_by_team("14-3 Girls")
        assert len(results) > 0
        assert all("14-3 Girls" in r["teams"] for r in results)

    def test_query_by_team_case_insensitive(self, schedule_loader):
        results_upper = schedule_loader.query_by_team("14-3 GIRLS")
        results_lower = schedule_loader.query_by_team("14-3 girls")
        assert len(results_upper) == len(results_lower)

    def test_query_by_team_partial(self, schedule_loader):
        # Substring match
        results = schedule_loader.query_by_team("14-3")
        assert len(results) > 0

    def test_query_by_team_with_date(self, schedule_loader):
        # Get first team and first date from data
        if schedule_loader.df.empty:
            pytest.skip("No schedule data")

        team = schedule_loader.df.iloc[0]["teams"]
        date_val = schedule_loader.df.iloc[0]["date"].date()

        results = schedule_loader.query_by_team(team, date_val)
        assert len(results) > 0
        assert all(r["teams"] == team for r in results)
        assert all(
            pd.to_datetime(r["date"]).date() == date_val for r in results
        )

    def test_query_by_court(self, schedule_loader):
        # Find courts that exist
        courts = schedule_loader.df["space"].unique()
        if len(courts) == 0:
            pytest.skip("No court data")

        court = courts[0]
        results = schedule_loader.query_by_court(str(court))
        assert len(results) > 0

    def test_query_by_date(self, schedule_loader):
        if schedule_loader.df.empty:
            pytest.skip("No schedule data")

        date_val = schedule_loader.df.iloc[0]["date"].date()
        results = schedule_loader.query_by_date(date_val)
        assert len(results) > 0
        assert all(
            pd.to_datetime(r["date"]).date() == date_val for r in results
        )

    def test_query_no_match(self, schedule_loader):
        results = schedule_loader.query_by_team("NonexistentTeam999")
        assert len(results) == 0

    def test_query_by_gender_without_classification(self, schedule_loader):
        # Should warn and return empty if cache not populated
        results = schedule_loader.query_by_gender("girls")
        assert len(results) == 0  # Cache is empty


class TestGenderClassification:
    """Test team gender classification."""

    @pytest.fixture
    def schedule_loader(self):
        docs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "docs"
        )
        xlsx_path = os.path.join(docs_dir, "CrossbarCourtSchedule.xlsx")
        return ScheduleLoader(xlsx_path)

    def test_get_all_team_names(self, schedule_loader):
        teams = schedule_loader.get_all_team_names()
        assert len(teams) > 0
        # Should be sorted
        assert teams == sorted(teams)

    def test_mock_gender_classification(self, schedule_loader):
        """Test with a mock LLM classifier."""
        # Define a mock classifier that uses simple heuristics
        def mock_classify(team_names):
            classification = {}
            for team in team_names:
                team_lower = team.lower()
                if "girl" in team_lower:
                    classification[team] = "girls"
                elif "boy" in team_lower:
                    classification[team] = "boys"
                else:
                    classification[team] = "other"
            return classification

        schedule_loader.classify_teams_by_gender(mock_classify)
        cache = schedule_loader.team_gender_cache

        assert len(cache) > 0
        # Should have classified at least some teams
        values = set(cache.values())
        assert "girls" in values or "boys" in values or "other" in values

    def test_query_by_gender_after_classification(self, schedule_loader):
        """Test gender-based queries after classification."""
        def mock_classify(team_names):
            classification = {}
            for team in team_names:
                team_lower = team.lower()
                if "girl" in team_lower:
                    classification[team] = "girls"
                elif "boy" in team_lower:
                    classification[team] = "boys"
                else:
                    classification[team] = "other"
            return classification

        schedule_loader.classify_teams_by_gender(mock_classify)
        results = schedule_loader.query_by_gender("girls")

        # Should return sessions for girls teams
        if len(results) > 0:
            assert all(
                schedule_loader.team_gender_cache.get(r["teams"]) == "girls"
                for r in results
            )


class TestScheduleFormatting:
    """Test schedule text formatting."""

    @pytest.fixture
    def schedule_loader(self):
        docs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "docs"
        )
        xlsx_path = os.path.join(docs_dir, "CrossbarCourtSchedule.xlsx")
        return ScheduleLoader(xlsx_path)

    def test_to_text_output(self, schedule_loader):
        text = schedule_loader.to_text()
        assert len(text) > 0
        assert "No Panic Volleyball Court Schedule" in text
        # Should have date headers
        assert "##" in text
        # Should have dashes for list items
        assert "-" in text
