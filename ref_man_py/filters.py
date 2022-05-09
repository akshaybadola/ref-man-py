from typing import Dict, List
import re


def year_filter(entry: Dict, min_y: int, max_y: int) -> bool:
    min_y = -1 if (min_y == "any" or not min_y) else min_y
    max_y = 10000 if (max_y == "any" or not max_y) else max_y
    return entry["year"] >= min_y and entry["year"] <= max_y


def author_filter(entry: Dict, author_names: List[str], author_ids: List[str]) -> bool:
    """Return True if any of the given authors by name or id are in the entry.

    Only one of ids or names are checked.

    """
    if author_ids:
        return any([a == x['authorId'] for a in author_ids for x in entry["authors"]])
    elif author_names:
        return any([a == x['name'] for a in author_names for x in entry["authors"]])
    else:
        return False


def num_citing_filter(entry: Dict, num: int) -> bool:
    """Return True if the number of citations is greater or equal than `num`"""
    return entry["citationCount"] >= num


def num_influential_count_filter(entry: Dict, num: int) -> bool:
    """Return True if the influential citations is greater or equal than `num`"""
    return entry["influentialCitationCount"] >= num


def venue_filter(entry: Dict, venues: List[str]) -> bool:
    """Return True if any of the given venues by regexp match are in the entry.

    The case in regexp match is ignored.

    """
    return any([re.match(x, entry["venue"], flags=re.IGNORECASE)
                for x in venues])


def title_filter(entry: Dict, title_re: str, invert: bool) -> bool:
    """Return True if the given regexp matches the entry title.

    The case in regexp match is ignored.
    Args:
        entry: A paper entry
        title_re: title regexp
        invert: Whether to include or exclude matching titles

    """
    match = bool(re.match(title_re, entry["title"], flags=re.IGNORECASE))
    return not match if invert else match
