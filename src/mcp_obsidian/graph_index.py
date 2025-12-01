"""
Graph Index Module

Provides a cached graph index for fast backlink queries and PageRank computation.
Builds the full vault graph lazily on first use and caches metadata for filtering.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from fnmatch import fnmatch
from datetime import datetime
from pathlib import Path

from .graph import LinkExtractor

logger = logging.getLogger(__name__)


class GraphIndex:
    """
    Pre-built graph index for fast backlink queries and PageRank scoring.

    Features:
    - Directed graph using dict-based adjacency lists
    - Cached PageRank scores (custom power iteration)
    - O(1) backlink/forward link lookups
    - Metadata cache (tags, frontmatter, dates) per note
    - Respects "Do Not Search" folder exclusions
    - Lazy initialization on first use
    """

    # Patterns for metadata extraction
    FRONTMATTER_PATTERN = re.compile(r'^---\n(.*?)\n---', re.DOTALL)
    TAG_PATTERN = re.compile(r'#([\w/-]+)')
    HEADING_PATTERN = re.compile(r'^#\s+(.+)$', re.MULTILINE)

    def __init__(self, file_getter: Callable[[str], str]):
        """
        Initialize the graph index.

        Args:
            file_getter: Callable that takes a file path and returns its content
        """
        self.file_getter = file_getter
        self.adjacency: Dict[str, Set[str]] = {}       # outgoing links
        self.reverse_adj: Dict[str, Set[str]] = {}     # incoming links (backlinks)
        self.pagerank_scores: Dict[str, float] = {}
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.exclude_paths: List[str] = []             # "Do Not Search" folders
        self.all_files: List[str] = []
        self._built = False
        self._build_time: Optional[datetime] = None

    def set_exclude_paths(self, patterns: List[str]) -> None:
        """
        Set folder patterns to exclude from indexing ("Do Not Search" folders).

        Args:
            patterns: List of glob patterns (e.g., ["Archive/*", "Templates/*"])
        """
        self.exclude_paths = patterns

    def _should_index(self, path: str) -> bool:
        """Check if path should be included in graph index."""
        for pattern in self.exclude_paths:
            if fnmatch(path, pattern):
                return False
        return True

    def build(self, all_files: List[str]) -> None:
        """
        Build full graph index from vault.

        Args:
            all_files: List of all file paths in the vault
        """
        start_time = datetime.now()
        logger.info(f"Building graph index from {len(all_files)} files...")

        # Reset state
        self.adjacency.clear()
        self.reverse_adj.clear()
        self.pagerank_scores.clear()
        self.metadata_cache.clear()
        self.all_files = all_files

        # Filter to markdown files and apply exclusions
        md_files = [
            f for f in all_files
            if isinstance(f, str) and f.endswith('.md') and self._should_index(f)
        ]

        # Build name-to-path lookup for link resolution
        name_to_path = self._build_name_to_path_map(md_files)

        # Process each file
        for file_path in md_files:
            try:
                content = self.file_getter(file_path)
                if not isinstance(content, str):
                    continue

                # Initialize adjacency for this node
                if file_path not in self.adjacency:
                    self.adjacency[file_path] = set()
                if file_path not in self.reverse_adj:
                    self.reverse_adj[file_path] = set()

                # Extract and cache metadata
                self.metadata_cache[file_path] = self._extract_metadata(file_path, content)

                # Extract outgoing links
                wikilinks = LinkExtractor.extract_wikilinks(content)
                for target, link_text, _ in wikilinks:
                    target_path = self._resolve_link(target, name_to_path)
                    if target_path and target_path != file_path:
                        # Add to adjacency (outgoing)
                        self.adjacency[file_path].add(target_path)

                        # Add to reverse adjacency (incoming/backlink)
                        if target_path not in self.reverse_adj:
                            self.reverse_adj[target_path] = set()
                        self.reverse_adj[target_path].add(file_path)

            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue

        # Compute PageRank
        self.compute_pagerank()

        self._built = True
        self._build_time = datetime.now()
        elapsed = (self._build_time - start_time).total_seconds()
        logger.info(
            f"Graph index built: {len(self.adjacency)} nodes, "
            f"{sum(len(v) for v in self.adjacency.values())} edges, "
            f"{elapsed:.2f}s"
        )

    def _build_name_to_path_map(self, files: List[str]) -> Dict[str, str]:
        """Build a map from normalized note names to file paths."""
        name_to_path = {}
        for file_path in files:
            # Normalize: lowercase, remove .md, extract filename
            normalized = file_path.lower()
            if normalized.endswith('.md'):
                normalized = normalized[:-3]
            filename = Path(normalized).name

            # Store both full path and filename for lookup
            name_to_path[normalized] = file_path
            name_to_path[filename] = file_path
            # Also handle spaces/underscores
            name_to_path[filename.replace(' ', '_')] = file_path
            name_to_path[filename.replace('_', ' ')] = file_path

        return name_to_path

    def _resolve_link(self, target: str, name_to_path: Dict[str, str]) -> Optional[str]:
        """Resolve a wikilink target to a file path."""
        if not target:
            return None

        # Normalize target
        normalized = target.lower().strip()
        if normalized.endswith('.md'):
            normalized = normalized[:-3]

        # Try direct lookup
        if normalized in name_to_path:
            return name_to_path[normalized]

        # Try filename only
        filename = Path(normalized).name
        if filename in name_to_path:
            return name_to_path[filename]

        # Try with space/underscore variants
        for variant in [filename.replace(' ', '_'), filename.replace('_', ' ')]:
            if variant in name_to_path:
                return name_to_path[variant]

        return None

    def _extract_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Extract metadata from file content.

        Returns dict with: tags, frontmatter, created, modified, title, word_count
        """
        metadata: Dict[str, Any] = {
            'tags': [],
            'frontmatter': {},
            'created': None,
            'modified': None,
            'title': Path(file_path).stem.replace('_', ' '),
            'word_count': len(content.split())
        }

        # Extract frontmatter
        match = self.FRONTMATTER_PATTERN.match(content)
        if match:
            yaml_str = match.group(1)
            frontmatter = self._parse_simple_yaml(yaml_str)
            metadata['frontmatter'] = frontmatter

            # Extract dates from frontmatter
            for date_key in ['created', 'date', 'dateCreated']:
                if date_key in frontmatter:
                    metadata['created'] = self._clean_date(frontmatter[date_key])
                    break

            for date_key in ['modified', 'updated', 'dateModified']:
                if date_key in frontmatter:
                    metadata['modified'] = self._clean_date(frontmatter[date_key])
                    break

            # Extract tags from frontmatter
            if 'tags' in frontmatter:
                fm_tags = frontmatter['tags']
                if isinstance(fm_tags, list):
                    metadata['tags'].extend(fm_tags)
                elif isinstance(fm_tags, str):
                    # Handle comma-separated or space-separated
                    metadata['tags'].extend([
                        t.strip().lstrip('#')
                        for t in re.split(r'[,\s]+', fm_tags)
                        if t.strip()
                    ])

        # Extract inline tags (outside frontmatter)
        content_after_fm = content
        if match:
            content_after_fm = content[match.end():]
        inline_tags = self.TAG_PATTERN.findall(content_after_fm)
        metadata['tags'].extend(inline_tags)
        metadata['tags'] = list(set(metadata['tags']))  # Dedupe

        # Extract title from first H1
        title_match = self.HEADING_PATTERN.search(content)
        if title_match:
            metadata['title'] = title_match.group(1).strip()

        return metadata

    def _parse_simple_yaml(self, yaml_str: str) -> Dict[str, Any]:
        """
        Simple YAML parser for frontmatter (key: value pairs).
        Does not handle nested structures or arrays.
        """
        result = {}
        for line in yaml_str.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                # Handle YAML lists (simple case: [item1, item2])
                if value.startswith('[') and value.endswith(']'):
                    value = [v.strip().strip('"').strip("'")
                             for v in value[1:-1].split(',') if v.strip()]
                result[key] = value
        return result

    def _clean_date(self, date_value: Any) -> Optional[str]:
        """Clean and normalize a date value to YYYY-MM-DD format."""
        if not date_value:
            return None

        date_str = str(date_value).strip()
        # Remove surrounding quotes
        date_str = date_str.strip('"').strip("'")

        # Try to extract YYYY-MM-DD
        match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
        if match:
            return match.group(1)

        return date_str if date_str else None

    def compute_pagerank(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> None:
        """
        Compute PageRank using power iteration (no external dependencies).

        Note: If performance issues arise with large vaults (>5000 notes),
        consider switching to NetworkX: `pip install networkx` and use
        `scores = nx.pagerank(G, alpha=0.85)` which is heavily optimized.

        Args:
            alpha: Damping factor (probability of following a link). Default 0.85.
            max_iter: Maximum iterations. Default 100.
            tol: Convergence tolerance. Default 1e-6.
        """
        nodes = list(self.adjacency.keys())
        n = len(nodes)

        if n == 0:
            self.pagerank_scores = {}
            return

        # Initialize with uniform distribution
        scores = {node: 1.0 / n for node in nodes}

        for iteration in range(max_iter):
            new_scores = {}

            for node in nodes:
                # Sum contributions from incoming links
                rank_sum = 0.0
                for src in self.reverse_adj.get(node, set()):
                    out_degree = len(self.adjacency.get(src, set()))
                    if out_degree > 0:
                        rank_sum += scores.get(src, 0) / out_degree

                # PageRank formula: (1-alpha)/n + alpha * sum(PR(src)/out_degree(src))
                new_scores[node] = (1 - alpha) / n + alpha * rank_sum

            # Check convergence
            diff = sum(abs(new_scores[node] - scores[node]) for node in nodes)
            scores = new_scores

            if diff < tol:
                logger.debug(f"PageRank converged after {iteration + 1} iterations")
                break

        self.pagerank_scores = scores

    def get_backlinks(self, path: str) -> List[str]:
        """
        Get all notes that link TO the given path (O(1) lookup).

        Args:
            path: File path to find backlinks for

        Returns:
            List of file paths that link to this note
        """
        return list(self.reverse_adj.get(path, set()))

    def get_forward_links(self, path: str) -> List[str]:
        """
        Get all notes that the given path links TO (O(1) lookup).

        Args:
            path: File path to find forward links for

        Returns:
            List of file paths this note links to
        """
        return list(self.adjacency.get(path, set()))

    def get_pagerank(self, path: str) -> float:
        """Get PageRank score for a note."""
        return self.pagerank_scores.get(path, 0.0)

    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get cached metadata for a note."""
        return self.metadata_cache.get(path, {})

    def is_built(self) -> bool:
        """Check if the index has been built."""
        return self._built

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        total_edges = sum(len(v) for v in self.adjacency.values())
        return {
            'total_nodes': len(self.adjacency),
            'total_edges': total_edges,
            'avg_out_degree': total_edges / len(self.adjacency) if self.adjacency else 0,
            'avg_pagerank': (
                sum(self.pagerank_scores.values()) / len(self.pagerank_scores)
                if self.pagerank_scores else 0
            ),
            'build_time': self._build_time.isoformat() if self._build_time else None,
            'excluded_patterns': self.exclude_paths
        }


# Global singleton for lazy initialization
_graph_index: Optional[GraphIndex] = None


def get_graph_index(file_getter: Callable[[str], str], all_files: List[str],
                    exclude_paths: Optional[List[str]] = None,
                    force_rebuild: bool = False) -> GraphIndex:
    """
    Get or create the global GraphIndex singleton.

    Args:
        file_getter: Callable to get file contents
        all_files: List of all file paths in vault
        exclude_paths: Optional list of glob patterns to exclude
        force_rebuild: If True, rebuild even if already built

    Returns:
        The GraphIndex instance
    """
    global _graph_index

    if _graph_index is None or force_rebuild:
        _graph_index = GraphIndex(file_getter)
        if exclude_paths:
            _graph_index.set_exclude_paths(exclude_paths)
        _graph_index.build(all_files)

    return _graph_index
