"""
Graph Filter Module

Provides filtering capabilities for graph traversal results.
Supports tags, frontmatter, path patterns, and date ranges.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from fnmatch import fnmatch
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """
    Filter configuration for graph traversal.

    All filters use AND logic - a node must pass ALL specified filters.
    """
    # Tag filtering
    tags: Optional[List[str]] = None
    tags_match_all: bool = False  # True=AND (all tags required), False=OR (any tag matches)

    # Frontmatter filtering
    frontmatter_filters: Optional[Dict[str, Any]] = None

    # Path filtering (glob patterns)
    include_paths: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None

    # Date filtering (YYYY-MM-DD format)
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    modified_after: Optional[str] = None
    modified_before: Optional[str] = None

    def has_filters(self) -> bool:
        """Check if any filters are configured."""
        return any([
            self.tags,
            self.frontmatter_filters,
            self.include_paths,
            self.exclude_paths,
            self.created_after,
            self.created_before,
            self.modified_after,
            self.modified_before
        ])

    def get_applied_filters(self) -> List[str]:
        """Return list of filter types that are active."""
        applied = []
        if self.tags:
            applied.append('tags')
        if self.frontmatter_filters:
            applied.append('frontmatter')
        if self.include_paths:
            applied.append('include_paths')
        if self.exclude_paths:
            applied.append('exclude_paths')
        if self.created_after or self.created_before:
            applied.append('created_date')
        if self.modified_after or self.modified_before:
            applied.append('modified_date')
        return applied


class GraphFilter:
    """
    Apply filters to graph nodes during traversal.

    Supports:
    - Tag filtering (any/all match modes)
    - Frontmatter field matching
    - Path patterns (include/exclude with glob syntax)
    - Date range filtering (created/modified)
    """

    # Date formats to try when parsing
    DATE_FORMATS = [
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d',
    ]

    def __init__(self, config: FilterConfig):
        """
        Initialize the filter with configuration.

        Args:
            config: FilterConfig specifying filter criteria
        """
        self.config = config

    def matches(self, path: str, metadata: Dict[str, Any]) -> bool:
        """
        Check if a node passes all configured filters.

        Args:
            path: File path of the node
            metadata: Metadata dict with tags, frontmatter, created, modified, etc.

        Returns:
            True if node passes all filters, False otherwise
        """
        # Path exclusion filter (check first for efficiency)
        if not self._check_path(path):
            return False

        # Tag filter
        if self.config.tags:
            if not self._check_tags(metadata.get('tags', [])):
                return False

        # Frontmatter filter
        if self.config.frontmatter_filters:
            if not self._check_frontmatter(metadata.get('frontmatter', {})):
                return False

        # Date filters
        if not self._check_dates(
            metadata.get('created'),
            metadata.get('modified')
        ):
            return False

        return True

    def _check_tags(self, node_tags: List[str]) -> bool:
        """
        Check tag filters with AND/OR logic.

        Args:
            node_tags: List of tags on the node (without # prefix)

        Returns:
            True if tag filter passes
        """
        if not self.config.tags:
            return True

        # Normalize tags (remove # prefix if present)
        filter_tags = [t.lstrip('#').lower() for t in self.config.tags]
        node_tags_normalized = [t.lstrip('#').lower() for t in node_tags]

        if self.config.tags_match_all:
            # AND logic: all filter tags must be present
            return all(tag in node_tags_normalized for tag in filter_tags)
        else:
            # OR logic: at least one filter tag must match
            return any(tag in node_tags_normalized for tag in filter_tags)

    def _check_frontmatter(self, frontmatter: Dict[str, Any]) -> bool:
        """
        Check frontmatter field filters.

        Supports simple equality matching on frontmatter fields.

        Args:
            frontmatter: Dict of frontmatter key-value pairs

        Returns:
            True if all frontmatter filters match
        """
        if not self.config.frontmatter_filters:
            return True

        for key, expected_value in self.config.frontmatter_filters.items():
            actual_value = frontmatter.get(key)

            # Handle None/missing
            if actual_value is None:
                return False

            # String comparison (case-insensitive)
            if isinstance(expected_value, str) and isinstance(actual_value, str):
                if expected_value.lower() != actual_value.lower():
                    return False
            # List membership (if expected is in actual list)
            elif isinstance(actual_value, list):
                if expected_value not in actual_value:
                    # Try case-insensitive string matching
                    if isinstance(expected_value, str):
                        if not any(
                            expected_value.lower() == str(v).lower()
                            for v in actual_value
                        ):
                            return False
                    else:
                        return False
            # Direct equality
            elif expected_value != actual_value:
                return False

        return True

    def _check_path(self, path: str) -> bool:
        """
        Check path include/exclude filters with glob patterns.

        Args:
            path: File path to check

        Returns:
            True if path passes filter
        """
        # Check exclude first (takes precedence)
        if self.config.exclude_paths:
            for pattern in self.config.exclude_paths:
                if fnmatch(path, pattern) or fnmatch(path, f"*/{pattern}"):
                    return False

        # Check include (if specified, path must match at least one)
        if self.config.include_paths:
            matches_include = any(
                fnmatch(path, pattern) or fnmatch(path, f"*/{pattern}")
                for pattern in self.config.include_paths
            )
            if not matches_include:
                return False

        return True

    def _check_dates(
        self,
        created: Optional[str],
        modified: Optional[str]
    ) -> bool:
        """
        Check date range filters.

        Args:
            created: Created date string (YYYY-MM-DD or similar)
            modified: Modified date string (YYYY-MM-DD or similar)

        Returns:
            True if date filters pass
        """
        # Check created date range
        if self.config.created_after or self.config.created_before:
            if not created:
                # No created date but filter requires it
                return False

            created_dt = self._parse_date(created)
            if not created_dt:
                return False

            if self.config.created_after:
                after_dt = self._parse_date(self.config.created_after)
                if after_dt and created_dt < after_dt:
                    return False

            if self.config.created_before:
                before_dt = self._parse_date(self.config.created_before)
                if before_dt and created_dt > before_dt:
                    return False

        # Check modified date range
        if self.config.modified_after or self.config.modified_before:
            if not modified:
                # No modified date but filter requires it
                return False

            modified_dt = self._parse_date(modified)
            if not modified_dt:
                return False

            if self.config.modified_after:
                after_dt = self._parse_date(self.config.modified_after)
                if after_dt and modified_dt < after_dt:
                    return False

            if self.config.modified_before:
                before_dt = self._parse_date(self.config.modified_before)
                if before_dt and modified_dt > before_dt:
                    return False

        return True

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse a date string into a datetime object.

        Args:
            date_str: Date string in various formats

        Returns:
            datetime object or None if parsing fails
        """
        if not date_str:
            return None

        # Clean the string
        date_str = str(date_str).strip().strip('"').strip("'")

        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try to extract just the date part
        try:
            # Handle "2024-01-15T10:30:00Z" or similar
            date_part = date_str.split('T')[0].split(' ')[0]
            return datetime.strptime(date_part, '%Y-%m-%d')
        except (ValueError, IndexError):
            pass

        logger.warning(f"Could not parse date: {date_str}")
        return None


def create_filter_from_args(args: Dict[str, Any]) -> Optional[FilterConfig]:
    """
    Create a FilterConfig from tool arguments.

    Args:
        args: Dict of tool arguments with filter_* keys

    Returns:
        FilterConfig if any filters specified, None otherwise
    """
    config = FilterConfig(
        tags=args.get('filter_tags'),
        tags_match_all=args.get('filter_tags_match_all', False),
        frontmatter_filters=args.get('filter_frontmatter'),
        include_paths=args.get('filter_include_paths'),
        exclude_paths=args.get('filter_exclude_paths'),
        created_after=args.get('filter_created_after'),
        created_before=args.get('filter_created_before'),
        modified_after=args.get('filter_modified_after'),
        modified_before=args.get('filter_modified_before'),
    )

    return config if config.has_filters() else None
