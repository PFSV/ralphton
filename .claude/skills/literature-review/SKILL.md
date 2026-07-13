---
name: literature-review
description: Survey the published literature on a topic and return a cited digest. Use when the user asks what prior work exists, whether an idea is novel, who else has done this, what a paper claims, or asks for related work for a paper section.
when_to_use: Trigger on "related work", "has anyone done", "is this novel", "what does the literature say", "find papers on", "cite prior work".
argument-hint: [topic or research question]
context: fork
agent: general-purpose
allowed-tools: mcp__arxiv__search_papers, mcp__arxiv__semantic_search, mcp__arxiv__get_abstract, mcp__arxiv__read_paper, mcp__semanticscholar__search_papers, mcp__semanticscholar__get_paper, mcp__semanticscholar__get_paper_citations, mcp__semanticscholar__get_paper_references, mcp__openalex__openalex_search_entities, mcp__openalex__openalex_analyze_trends, WebFetch, Read, Grep, Glob
---

Survey the literature on: $ARGUMENTS

Search, do not recall. A paper you cannot retrieve does not exist.

1. Search arXiv and Semantic Scholar for the topic. Search OpenAlex for adjacent framings the first two miss.
2. For the papers that matter, follow citations both ways: who they cite, who cites them. A field's real structure is in the citation graph, not the search ranking.
3. Read abstracts. Read the method section of any paper whose result the user's work depends on or contradicts.
4. Note where the literature disagrees with itself. Disagreement is the interesting signal.

Return:

- **Digest** — what is established, what is contested, what is unexamined.
- **Closest prior work** — the 3-8 papers that most constrain the user's question, each with venue, year, and one line on what it actually shows (not what its title promises).
- **Gap** — the specific claim the user's work would add, stated so that it could be wrong.
- **Bibliography** — every paper cited above, with its identifier (arXiv id or DOI).

Every claim in your output carries a citation. If you could not retrieve a source for a statement, delete the statement. Do not describe a paper you did not open.
