"""
Graph Ranker Module

Provides ranking capabilities for graph traversal results using
PageRank and recency weighting, aligned with smart search P1 configuration.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RankingConfig:
    """
    Ranking configuration for graph results.

    Defaults are aligned with smart search P1_RecencyBoost (768d optimized):
    - recency_weight: 0.340 (34% recency in final blend)
    - decay_days: 10.25 (~10 day exponential half-life)
    """
    enable_pagerank: bool = True
    enable_recency: bool = True
    recency_weight: float = 0.340        # From P1_RecencyBoost
    decay_days: float = 10.25            # From P1_RecencyBoost
    pagerank_alpha: float = 0.85         # Standard PageRank damping


class GraphRanker:
    """
    Rank graph results using PageRank + recency weighting.

    Algorithm (aligned with smart search P1_RecencyBoost):
    1. pagerank_norm = minmax_normalize(pagerank_scores)
    2. recency = exp(-days_old / decay_days)
    3. combined = (1 - recency_weight) * pagerank_norm + recency_weight * recency

    This provides a balanced ranking that considers both:
    - Structural importance (PageRank - how well-connected/authoritative)
    - Temporal relevance (Recency - how recently created/modified)
    """

    # Date formats to try when parsing
    DATE_FORMATS = [
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d',
    ]

    def __init__(self, config: Optional[RankingConfig] = None):
        """
        Initialize the ranker.

        Args:
            config: RankingConfig or None for defaults
        """
        self.config = config or RankingConfig()

    def rank(
        self,
        nodes: List[Dict[str, Any]],
        pagerank_scores: Dict[str, float],
        sort_by: str = 'relevance'
    ) -> List[Dict[str, Any]]:
        """
        Rank nodes by PageRank + recency.

        Args:
            nodes: List of node dicts with 'path', 'created', etc.
            pagerank_scores: Dict mapping path -> PageRank score
            sort_by: Sort mode - 'relevance', 'recency', 'pagerank', 'hop_distance'

        Returns:
            List of nodes with added score fields, sorted by specified criteria
        """
        if not nodes:
            return nodes

        # Add scores to each node
        for node in nodes:
            path = node.get('path', '')

            # PageRank score
            pagerank = pagerank_scores.get(path, 0.0)
            node['pagerank'] = pagerank

            # Recency score
            created = node.get('created') or node.get('metadata', {}).get('created')
            recency = self._compute_recency_score(created) if self.config.enable_recency else 0.0
            node['recency_score'] = recency

        # Normalize PageRank scores across result set
        if self.config.enable_pagerank:
            pagerank_values = [n['pagerank'] for n in nodes]
            normalized_pr = self._minmax_normalize(pagerank_values)
            for i, node in enumerate(nodes):
                node['pagerank_normalized'] = normalized_pr[i]
        else:
            for node in nodes:
                node['pagerank_normalized'] = 0.0

        # Compute combined score
        for node in nodes:
            if self.config.enable_pagerank and self.config.enable_recency:
                # Blend PageRank and recency
                pr_weight = 1.0 - self.config.recency_weight
                combined = (
                    pr_weight * node['pagerank_normalized'] +
                    self.config.recency_weight * node['recency_score']
                )
            elif self.config.enable_pagerank:
                combined = node['pagerank_normalized']
            elif self.config.enable_recency:
                combined = node['recency_score']
            else:
                combined = 0.0

            node['combined_score'] = combined

        # Sort by specified criteria
        return self._sort_nodes(nodes, sort_by)

    def _compute_recency_score(self, created_date: Optional[str]) -> float:
        """
        Compute recency score using exponential decay.

        Score = exp(-days_old / decay_days)

        With decay_days=10.25:
        - Today: 1.0
        - 10 days ago: ~0.37
        - 30 days ago: ~0.05
        - 90 days ago: ~0.00

        Args:
            created_date: Date string (YYYY-MM-DD or similar)

        Returns:
            Recency score between 0 and 1
        """
        if not created_date:
            # No date = neutral score (middle of range)
            return 0.5

        created_dt = self._parse_date(created_date)
        if not created_dt:
            return 0.5

        # Calculate days since creation
        now = datetime.now()
        days_old = (now - created_dt).days

        # Exponential decay
        if days_old < 0:
            # Future date (shouldn't happen, but handle gracefully)
            return 1.0

        score = math.exp(-days_old / self.config.decay_days)
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

    def _minmax_normalize(self, values: List[float]) -> List[float]:
        """
        Min-max normalize values to [0, 1] range.

        Args:
            values: List of float values

        Returns:
            Normalized values
        """
        if not values:
            return []

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            # All same value - return 0.5 for all
            return [0.5] * len(values)

        return [(v - min_val) / (max_val - min_val) for v in values]

    def _sort_nodes(
        self,
        nodes: List[Dict[str, Any]],
        sort_by: str
    ) -> List[Dict[str, Any]]:
        """
        Sort nodes by specified criteria.

        Args:
            nodes: List of nodes with score fields
            sort_by: Sort mode

        Returns:
            Sorted list of nodes
        """
        if sort_by == 'relevance':
            # Combined score (PageRank + recency), descending
            return sorted(nodes, key=lambda n: n.get('combined_score', 0), reverse=True)

        elif sort_by == 'recency':
            # Recency score, descending (most recent first)
            return sorted(nodes, key=lambda n: n.get('recency_score', 0), reverse=True)

        elif sort_by == 'pagerank':
            # PageRank score, descending (most important first)
            return sorted(nodes, key=lambda n: n.get('pagerank', 0), reverse=True)

        elif sort_by == 'hop_distance':
            # Hop distance, ascending (closest first)
            return sorted(nodes, key=lambda n: n.get('hop_distance', 999))

        else:
            # Default to relevance
            logger.warning(f"Unknown sort_by value: {sort_by}, using relevance")
            return sorted(nodes, key=lambda n: n.get('combined_score', 0), reverse=True)

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
            date_part = date_str.split('T')[0].split(' ')[0]
            return datetime.strptime(date_part, '%Y-%m-%d')
        except (ValueError, IndexError):
            pass

        return None


def create_ranker_from_args(args: Dict[str, Any]) -> GraphRanker:
    """
    Create a GraphRanker from tool arguments.

    Args:
        args: Dict of tool arguments

    Returns:
        Configured GraphRanker
    """
    config = RankingConfig(
        enable_pagerank=args.get('enable_pagerank', True),
        enable_recency=args.get('enable_recency', True),
        # Use defaults for weights (aligned with P1_RecencyBoost)
    )

    return GraphRanker(config)
