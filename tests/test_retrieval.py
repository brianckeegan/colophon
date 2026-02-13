import unittest

from colophon.models import Source
from colophon.retrieval import SimpleRetriever


class RetrievalTests(unittest.TestCase):
    def test_search_returns_ranked_hits_and_respects_top_k(self) -> None:
        sources = [
            Source(id="s1", title="One", authors=[], year=None, text="alpha beta gamma"),
            Source(id="s2", title="Two", authors=[], year=None, text="alpha beta"),
            Source(id="s3", title="Three", authors=[], year=None, text="delta epsilon"),
        ]
        retriever = SimpleRetriever(sources)

        hits = retriever.search("alpha beta gamma", top_k=2)

        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].source.id, "s1")
        self.assertEqual(hits[1].source.id, "s2")
        self.assertGreaterEqual(hits[0].score, hits[1].score)

    def test_search_returns_empty_for_no_overlap(self) -> None:
        sources = [Source(id="s1", title="One", authors=[], year=None, text="apple orange")]
        retriever = SimpleRetriever(sources)

        hits = retriever.search("banana", top_k=5)

        self.assertEqual(hits, [])

    def test_search_returns_empty_for_empty_query(self) -> None:
        sources = [Source(id="s1", title="One", authors=[], year=None, text="apple orange")]
        retriever = SimpleRetriever(sources)

        hits = retriever.search("", top_k=5)

        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
