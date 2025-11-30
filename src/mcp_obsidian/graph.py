import re
from typing import Optional, Set, Dict, List, Tuple
from collections import deque
from pathlib import Path


class LinkExtractor:
    """Extract wikilinks and markdown links from note content."""

    # Pattern for [[note]] or [[note|display]]
    WIKILINK_PATTERN = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')

    # Pattern for [text](path) markdown links
    MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    @staticmethod
    def extract_wikilinks(content: str) -> List[Tuple[str, str, int]]:
        """Extract all wikilinks from content.

        Returns:
            List of (target_path, link_text, position) tuples
        """
        links = []
        for match in LinkExtractor.WIKILINK_PATTERN.finditer(content):
            target = match.group(1).strip()
            # Remove heading anchors (#)
            if '#' in target:
                target = target.split('#')[0]
            # Normalize path
            target = target.strip()
            link_text = match.group(2) or target
            links.append((target, link_text, match.start()))
        return links

    @staticmethod
    def extract_markdown_links(content: str) -> List[Tuple[str, str, int]]:
        """Extract all markdown links from content.

        Returns:
            List of (target_path, link_text, position) tuples
        """
        links = []
        for match in LinkExtractor.MARKDOWN_LINK_PATTERN.finditer(content):
            link_text = match.group(1)
            target = match.group(2)
            # Skip http/https links
            if target.startswith(('http://', 'https://')):
                continue
            links.append((target, link_text, match.start()))
        return links


class LinkContextExtractor:
    """Extract surrounding context for a link."""

    @staticmethod
    def extract_context(
        content: str,
        target_path: str,
        window_chars: int = 150
    ) -> str:
        """Extract context around a link to a target note.

        Args:
            content: Note content
            target_path: Path we're looking for links to
            window_chars: Characters to include on each side of link

        Returns:
            Context string with the link in the middle
        """
        # Find first occurrence of the target
        links = LinkExtractor.extract_wikilinks(content)

        best_context = ""
        best_position = -1

        for target, link_text, position in links:
            # Normalize comparison
            if target.lower() == target_path.lower() or \
               target.lower().replace(' ', '_') == target_path.lower().replace(' ', '_'):
                if best_position == -1 or position < best_position:
                    best_position = position

                    # Extract surrounding context with sentence boundaries
                    start = max(0, position - window_chars)
                    end = min(len(content), position + len(f"[[{target}]]") + window_chars)

                    # Try to snap to sentence boundaries
                    context_start = start
                    context_end = end

                    # Find previous sentence boundary
                    for i in range(position - 1, start, -1):
                        if content[i] in '.!?' and i + 1 < len(content) and content[i + 1].isspace():
                            context_start = i + 2
                            break

                    # Find next sentence boundary
                    for i in range(end, position, -1):
                        if i < len(content) and content[i] in '.!?' and \
                           (i + 1 >= len(content) or content[i + 1].isspace()):
                            context_end = i + 1
                            break

                    best_context = content[context_start:context_end].strip()

        # Fallback to first N chars if no link found
        if not best_context:
            best_context = content[:window_chars * 2].strip()

        return best_context


class NoteGraph:
    """Build and traverse a note graph."""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str, str]] = []
        self.visited: Set[str] = set()

    def normalize_path(self, path: str) -> str:
        """Normalize note paths for comparison."""
        # Remove leading/trailing slashes and .md extension
        path = path.strip('/').lower()
        if path.endswith('.md'):
            path = path[:-3]
        return path

    def find_note_path(self, target: str, all_files: List[str]) -> Optional[str]:
        """Find matching note path from target name.

        Args:
            target: The note name/path to find
            all_files: List of all file paths in vault

        Returns:
            The full path to the matching file, or None
        """
        if not target or not all_files:
            return None

        target_normalized = self.normalize_path(target)
        target_filename = Path(target_normalized).name

        for file_path in all_files:
            if not isinstance(file_path, str):
                continue

            normalized = self.normalize_path(file_path)

            # Direct match
            if normalized == target_normalized:
                return file_path

            # Match just the filename (most common case for wikilinks)
            filename = Path(normalized).name
            if filename == target_filename:
                return file_path

            # Handle notes with spaces vs underscores
            filename_underscores = filename.replace(' ', '_')
            target_underscores = target_filename.replace(' ', '_')
            if filename_underscores == target_underscores:
                return file_path

        return None

    def add_note(self, path: str, title: str, word_count: int, last_modified: str):
        """Add a note to the graph."""
        self.nodes[path] = {
            'path': path,
            'title': title,
            'word_count': word_count,
            'last_modified': last_modified
        }

    def add_edge(self, from_path: str, to_path: str, link_type: str, context: str = ""):
        """Add an edge between two notes."""
        self.edges.append((from_path, to_path, link_type))
        if 'neighbors' not in self.nodes[from_path]:
            self.nodes[from_path]['neighbors'] = []
        self.nodes[from_path]['neighbors'].append({
            'to': to_path,
            'type': link_type,
            'context': context
        })

    def traverse(
        self,
        start_path: str,
        all_files: List[str],
        file_getter: callable,
        max_hops: int = 1,
        direction: str = "both",
        max_nodes: int = 30
    ) -> Dict:
        """Traverse the graph from a starting note.

        Args:
            start_path: Path to start traversal from
            all_files: List of all files in vault (file paths as strings)
            file_getter: Callable to get file contents
            max_hops: Maximum number of hops (1 or 2)
            direction: "outgoing", "incoming", or "both"
            max_nodes: Maximum number of nodes to return

        Returns:
            Dict with nodes, edges, and metadata
        """
        self.visited.clear()
        queue = deque([(start_path, 0)])  # (path, hop_distance)
        result_nodes = {}
        result_edges = []

        # Build a map of note names to file paths for faster lookup
        note_name_to_path = {}
        for file_path in all_files:
            if isinstance(file_path, str):
                normalized = self.normalize_path(file_path)
                filename = Path(normalized).name
                note_name_to_path[filename] = file_path

        while queue and len(result_nodes) < max_nodes:
            current_path, hop = queue.popleft()

            if current_path in self.visited:
                continue

            self.visited.add(current_path)

            # Get file content
            try:
                content = file_getter(current_path)
                if not isinstance(content, str):
                    continue
            except Exception as e:
                continue

            # Extract title from first heading or filename
            title = self._extract_title(content, current_path)
            word_count = len(content.split())

            # Add node
            node_data = {
                'path': current_path,
                'title': title,
                'word_count': word_count,
                'snippet': '',
                'hop_distance': hop
            }
            result_nodes[current_path] = node_data

            # Find links based on direction
            if hop < max_hops:
                # Outgoing links: links FROM current note
                if direction in ("outgoing", "both"):
                    try:
                        outgoing = LinkExtractor.extract_wikilinks(content)
                    except Exception:
                        outgoing = []

                    for target, link_text, _ in outgoing:
                        target_path = self.find_note_path(target, all_files)
                        if target_path and target_path not in self.visited:
                            result_edges.append({
                                'from': current_path,
                                'to': target_path,
                                'type': 'wikilink',
                                'link_text': link_text
                            })
                            queue.append((target_path, hop + 1))

                # Incoming links: links TO current note (backlinks)
                if direction in ("incoming", "both"):
                    # Find all files that link to the current note
                    current_filename = Path(self.normalize_path(current_path)).name

                    for file_path in all_files:
                        if file_path == current_path or file_path in self.visited:
                            continue

                        try:
                            file_content = file_getter(file_path)
                            if not isinstance(file_content, str):
                                continue

                            # Extract wikilinks from this file
                            links = LinkExtractor.extract_wikilinks(file_content)
                            found_link = False
                            for target, link_text, _ in links:
                                # Check if this file links to current note
                                target_filename = Path(self.normalize_path(target)).name
                                if target_filename == current_filename:
                                    if file_path not in result_nodes:
                                        result_edges.append({
                                            'from': file_path,
                                            'to': current_path,
                                            'type': 'wikilink',
                                            'link_text': link_text
                                        })
                                        queue.append((file_path, hop + 1))
                                        found_link = True
                                        break
                        except Exception:
                            continue

        return {
            'center_node': {
                'path': start_path,
                'title': result_nodes.get(start_path, {}).get('title', '')
            },
            'nodes': list(result_nodes.values()),
            'edges': result_edges,
            'summary': {
                'total_nodes': len(result_nodes),
                'total_edges': len(result_edges),
                'max_hops': max_hops
            }
        }

    @staticmethod
    def _extract_title(content: str, filepath: str) -> str:
        """Extract title from content or filepath."""
        # Try to find first H1 heading
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Fall back to filename
        return Path(filepath).stem.replace('_', ' ')
