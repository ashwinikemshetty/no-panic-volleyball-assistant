import os
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta, date


def resolve_date(date_expr: Optional[str]) -> Optional[date]:
    """Resolve date expressions to actual dates.

    Handles: "today", "tomorrow", "yesterday", "05/27/2026", "27/05/2026", "May 27", etc.

    Args:
        date_expr: User-provided date string or None

    Returns:
        Resolved date object, or None if no date mentioned or parse fails
    """
    if not date_expr or not str(date_expr).strip():
        return None

    date_expr = str(date_expr).strip().lower()
    today = datetime.now().date()

    # Handle relative dates
    if date_expr == "today":
        return today
    elif date_expr == "tomorrow":
        return today + timedelta(days=1)
    elif date_expr == "yesterday":
        return today - timedelta(days=1)

    # Try standard date parsing (MM/DD/YYYY, DD/MM/YYYY, Month Day Year, etc.)
    # Try MM/DD/YYYY first (US format)
    try:
        parsed = pd.to_datetime(date_expr, format="%m/%d/%Y", errors="raise")
        return parsed.date()
    except:
        pass

    # Try DD/MM/YYYY (non-US format)
    try:
        parsed = pd.to_datetime(date_expr, format="%d/%m/%Y", errors="raise")
        return parsed.date()
    except:
        pass

    # Try flexible parsing (Month Day, Month Day Year, etc.)
    try:
        parsed = pd.to_datetime(date_expr, errors="raise")
        # If year not specified, assume current year
        if "," not in date_expr and date_expr.count("/") == 0:
            parsed = parsed.replace(year=today.year)
        return parsed.date()
    except:
        pass

    # Last resort: check if it's something like "27th" and return nearest date
    try:
        # Extract day number if it ends with th/st/nd/rd
        day_str = date_expr.rstrip("stndrh")  # Remove ordinal suffixes
        day = int(day_str)
        if 1 <= day <= 31:
            # Return nearest matching day (this month or next)
            candidate = today.replace(day=day)
            if candidate < today:
                # Move to next month
                if today.month == 12:
                    candidate = candidate.replace(year=today.year + 1, month=1)
                else:
                    candidate = candidate.replace(month=today.month + 1)
            return candidate
    except:
        pass

    return None


class ScheduleLoader:
    """Load and query crossbar volleyball schedule data."""

    def __init__(self, xlsx_path: str):
        """Load schedule from XLSX file.

        Args:
            xlsx_path: Path to CrossbarCourtSchedule.xlsx
        """
        self.df = None
        self.team_gender_cache = {}
        self.load_schedule(xlsx_path)

    def load_schedule(self, xlsx_path: str) -> None:
        """Load XLSX into DataFrame and normalize."""
        if not os.path.exists(xlsx_path):
            print(f"WARNING: Schedule file not found: {xlsx_path}")
            return

        try:
            self.df = pd.read_excel(xlsx_path)
            # Normalize column names (case-insensitive, spaces to underscores)
            self.df.columns = [col.lower().replace(" ", "_") for col in self.df.columns]
            # Ensure dates are datetime objects
            if "date" in self.df.columns:
                self.df["date"] = pd.to_datetime(self.df["date"])
            print(f"✓ Loaded schedule: {len(self.df)} rows")
        except Exception as e:
            print(f"ERROR loading schedule: {e}")

    def get_all_team_names(self) -> List[str]:
        """Get list of all unique team names in schedule.

        Returns:
            Sorted list of unique team names
        """
        if self.df is None or self.df.empty:
            return []
        # Filter out NaN values
        teams = [t for t in self.df["teams"].unique() if pd.notna(t)]
        return sorted(teams)

    def classify_teams_by_gender(self, llm_classify_func) -> Dict[str, str]:
        """Use LLM to classify each team as girls/boys/mixed/other.

        Args:
            llm_classify_func: Function that takes team_names list and returns
                              dict {team_name: gender_label}

        Returns:
            Dict mapping team_name → "girls" | "boys" | "mixed" | "other"
        """
        team_names = self.get_all_team_names()
        if not team_names:
            return {}

        # Call LLM classification
        self.team_gender_cache = llm_classify_func(team_names)
        return self.team_gender_cache

    def query_by_team(self, team_name: str, date_val: Optional[date] = None) -> List[Dict]:
        """Find all sessions for a specific team, optionally on a date.

        Args:
            team_name: Team/session name (case-insensitive substring match)
            date_val: Optional date filter (date object or None)

        Returns:
            List of matching rows as dicts
        """
        if self.df is None or self.df.empty:
            return []

        matches = self.df[self.df["teams"].str.contains(team_name, case=False, na=False, regex=False)]

        if date_val is not None:
            date_val = pd.Timestamp(date_val).date() if not isinstance(date_val, date) else date_val
            matches = matches[pd.to_datetime(matches["date"]).dt.date == date_val]

        return sorted(matches.to_dict("records"), key=lambda x: x.get("start", ""))

    def query_by_gender(self, gender_label: str, date_val: Optional[date] = None) -> List[Dict]:
        """Find all sessions matching a gender classification.

        Args:
            gender_label: "girls" | "boys" | "mixed" (uses cached classification)
            date_val: Optional date filter

        Returns:
            List of matching rows as dicts
        """
        if self.df is None or self.df.empty:
            return []

        if not self.team_gender_cache:
            print("WARNING: Team gender cache not populated. Call classify_teams_by_gender() first.")
            return []

        # Filter teams that match the gender label
        matching_teams = [
            team for team, gender in self.team_gender_cache.items()
            if gender.lower() == gender_label.lower()
        ]

        if not matching_teams:
            return []

        # Filter df to rows where teams are in matching_teams
        matches = self.df[self.df["teams"].isin(matching_teams)]

        if date_val is not None:
            date_val = pd.Timestamp(date_val).date() if not isinstance(date_val, date) else date_val
            matches = matches[pd.to_datetime(matches["date"]).dt.date == date_val]

        return sorted(matches.to_dict("records"), key=lambda x: x.get("start", ""))

    def query_by_court(self, court: str, date_val: Optional[date] = None) -> List[Dict]:
        """Find all sessions on a specific court, optionally on a date.

        Args:
            court: Court name (case-insensitive substring match)
            date_val: Optional date filter

        Returns:
            List of matching rows as dicts
        """
        if self.df is None or self.df.empty:
            return []

        matches = self.df[self.df["space"].str.contains(court, case=False, na=False, regex=False)]

        if date_val is not None:
            date_val = pd.Timestamp(date_val).date() if not isinstance(date_val, date) else date_val
            matches = matches[pd.to_datetime(matches["date"]).dt.date == date_val]

        return sorted(matches.to_dict("records"), key=lambda x: x.get("start", ""))

    def query_by_date(self, date_val: date) -> List[Dict]:
        """Get all sessions on a specific date.

        Args:
            date_val: Date object

        Returns:
            List of all sessions on that date, sorted by start time
        """
        if self.df is None or self.df.empty:
            return []

        date_val = pd.Timestamp(date_val).date() if not isinstance(date_val, date) else date_val
        matches = self.df[pd.to_datetime(self.df["date"]).dt.date == date_val]
        return sorted(matches.to_dict("records"), key=lambda x: x.get("start", ""))

    def to_text(self) -> str:
        """Convert schedule to readable text format for debugging/review.

        Returns:
            Formatted schedule text
        """
        if self.df is None or self.df.empty:
            return ""

        lines = ["# No Panic Volleyball Court Schedule\n"]

        # Group by date for readability
        for date_val in sorted(self.df["date"].unique()):
            date_str = pd.to_datetime(date_val).strftime("%A, %B %d, %Y")
            lines.append(f"\n## {date_str}")

            day_schedule = self.df[self.df["date"] == date_val].sort_values("start")

            for _, row in day_schedule.iterrows():
                teams = row.get("teams", "N/A")
                court = row.get("space", "N/A")
                start = row.get("start", "N/A")
                end = row.get("end", "N/A")

                lines.append(f"- **{teams}** | {court} | {start} - {end}")

        return "\n".join(lines)
