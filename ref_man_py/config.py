from typing import Optional
import dataclasses
from dataclasses import dataclass, field


def default_fields():
    return {"search": {"limit": 10,
                       "fields": ['authors', 'abstract', 'title',
                                  'venue', 'publicationVenue', 'paperId', 'year',
                                  'url', 'citationCount',
                                  'influentialCitationCount',
                                  'externalIds']},
            "details": {"limit": 100,
                        "fields": ['authors', 'abstract', 'title',
                                   'venue', 'publicationVenue', 'paperId', 'year',
                                   'url', 'citationCount',
                                   'influentialCitationCount',
                                   'externalIds']},
            "citations": {"limit": 100,
                          "fields": ['authors', 'abstract', 'title',
                                     'venue', 'publicationVenue', 'paperId', 'year',
                                     'contexts', 'url', 'citationCount',
                                     'influentialCitationCount',
                                     'externalIds']},
            "references": {"limit": 100,
                           "fields": ['authors', 'abstract', 'title',
                                      'venue', 'publicationVenue', 'paperId', 'year',
                                      'contexts', 'url', 'citationCount',
                                      'influentialCitationCount',
                                      'externalIds']},
            "author": {"limit": 100,
                       "fields": ["authorId", "name"]},
            "author_papers": {"limit": 100,
                              "fields": ['authors', 'abstract', 'title',
                                         'venue', 'publicationVenue', 'paperId', 'year',
                                         'url', 'citationCount',
                                         'influentialCitationCount',
                                         'externalIds']}}


@dataclass
class DataParams:
    limit: int
    fields: list[str | list[str]] = field(default_factory=list)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __getitem__(self, k):
        return getattr(self, k)


@dataclass
class DataConfig:
    details: DataParams
    references: DataParams
    citations: DataParams
    search: DataParams
    author: DataParams
    author_papers: DataParams

    def __setattr__(self, k, v):
        super().__setattr__(k, DataParams(**v))

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __getitem__(self, k):
        return getattr(self, k)


@dataclass
class Config:
    """The configuration dataclass for :mod:`ref_man_py`

    This class defins both the API config and :class:`s2cache.SemanticScholar` config.

    Args:
        s2: dict
        cache_dir: str
        data: DataConfig
        api_key: Optional[str] = None
        batch_size: int = 500
        client_timeout: int = 10
        cache_backend: str = jsonl
        corpus_cache_dir: Optional[str] = None

    """
    data: DataConfig
    s2: dict = field(default_factory=dict)

    def __post_init__(self):
        self._keys = ["s2", "data"]
        if set([x.name for x in dataclasses.fields(self)]) != set(self._keys):
            raise AttributeError("self._keys should be same as fields")

    def __setattr__(self, k, v):
        if k == "data":
            super().__setattr__(k, DataConfig(**v))
        else:
            super().__setattr__(k, v)

    def __iter__(self):
        return iter(self._keys)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __getitem__(self, k):
        return getattr(self, k)
