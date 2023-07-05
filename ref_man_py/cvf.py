from typing import List, Dict, Optional, Tuple, Union
import os
import re
import operator
from pathlib import Path

import bs4
from bs4 import BeautifulSoup
import requests


class CVF:
    def __init__(self, files_dir, logger):
        """A CVF class to manage CVF links.

        Caches and manages links from open access CVF paper repositories so
        as not to burden it with excessive requests and save time. The
        downloaded html pages are kept in a :code:`files_dir` and pdf links
        extracted for easy search.

        Args:
            files_dir: Files directory. This is where all the files
                       would be kept
            logger: A :class:`logging.Logger` instance

        """
        self.files_dir = files_dir
        self.logger = logger
        self.cvf_url_root = "https://openaccess.thecvf.com"
        self.soups: Dict[Tuple, bs4.Tag] = {}
        self.cvf_pdf_links: Dict[Tuple, List[str]] = {}
        self._requests_timeout = 5
        # self.load_cvf_files()
        self.load_cvf_pdf_links()

    def _get(self, url: str, **kwargs) -> requests.Response:
        """Get a :code:`url` with sensible defaults

        Args:
            url: The url to fetch
            kwargs: kwargs to pass on to :meth:`requests.get`

        """
        return requests.get(url, timeout=self._requests_timeout, **kwargs)

    def get_pdf_link(self, title: str, venue: str, year: Optional[str]) -> Optional[str]:
        """Get a pdf link if it exists from the CVF files.
        The links are searched with given :code:`title`, :code:`venue` and :code:`year`
        Although :code:`year` is optional, it's better to give it for faster search.

        Args:
            title: Title of the paper
            venue: Venue where it appeared
            year: Optional year

        Returns:
            A string of title;url if found, otherwise None

        """
        venue = venue.lower()
        if year:
            keys = [(v, y) for v, y in self.cvf_pdf_links
                    if v == venue and y == year]
        else:
            keys = [(v, y) for v, y in self.cvf_pdf_links
                    if v == venue]
            year = ",".join([y for v, y in self.soups
                             if v == venue])
        if not keys and year:
            self.logger.debug(f"Fetching page(s) for {venue.upper()}{year}")
            self.download_cvf_page_and_update_soups(venue, year)
            self.save_cvf_pdf_links_and_update(venue, year, self.soups[(venue, year)])
            keys = [(v, y) for v, y in self.soups if v == venue and y == year]
        # maybe_link = self.find_soup(keys, title)
        return self.find_pdf_link(keys, title)

    def read_cvf_pdf_links(self, venue: str, year: str) -> List[str]:
        fname = self.files_dir.joinpath(f"{venue.upper()}{year}_pdfs")
        with open(fname) as f:
            pdf_links = f.read().split("\n")
        return pdf_links

    def load_cvf_files(self):
        """Load the CVF Soups from HTML files.

        XML parses via :class:`BeautifulSoup` are maintained for easy
        fetching of an article in case it's availble.

        """
        self.cvf_files = [os.path.join(self.files_dir, f)
                          for f in os.listdir(self.files_dir)
                          if re.match(r'^(cvpr|iccv)', f.lower())
                          and not f.endswith("_pdfs")]
        self.logger.debug("Loading CVF soups.")
        for cvf in self.cvf_files:
            if not cvf.endswith("_pdfs"):
                match = re.match(r'^(cvpr|iccv)(.*?)([0-9]+)',
                                 Path(cvf).name, flags=re.IGNORECASE)
                if match:
                    venue, _, year = map(str.lower, match.groups())
                    with open(cvf) as f:
                        self.soups[(venue, year)] = BeautifulSoup(f.read(), features="lxml")
                else:
                    self.logger.error(f"Could not load file {cvf}")
        self.logger.debug(f"Loaded conference files {self.soups.keys()}")

    def load_cvf_pdf_links(self):
        """Load the CVF PDF links saved from HTML files.

        """
        self.cvf_pdf_link_files = [os.path.join(self.files_dir, f)
                                   for f in os.listdir(self.files_dir)
                                   if re.match(r'^(cvpr|iccv)', f.lower())
                                   and f.endswith("_pdfs")]
        self.logger.debug("Loading CVF pdf links.")
        for fname in self.cvf_pdf_link_files:
            match = re.match(r'^(cvpr|iccv)(.*?)([0-9]+)',
                             Path(fname).name, flags=re.IGNORECASE)
            if match:
                venue, _, year = map(str.lower, match.groups())
                with open(fname) as f:
                    self.cvf_pdf_links[(venue, year)] = f.read().split("\n")
            else:
                self.logger.error(f"Could not load pdf links from {fname}")
        self.logger.debug(f"Loaded PDF links {self.cvf_pdf_links.keys()}")

    def best_match(self, title: str, matches: List) -> str:
        """Subroutine for finding the best match from regexp matches

        The match with the longest regexp match span is returned

        Args:
            title: Title to match
            matches: List of regexp matches


        """
        if not matches:
            return f"URL Not found for {title}"
        elif len(matches) == 1:
            href = matches[0].group(0)
        else:
            matches.sort(key=lambda x: operator.abs(operator.sub(*x.span())))
            href = matches[-1].group(0)
        href = "https://openaccess.thecvf.com/" + href.lstrip("/")
        return f"{title};{href}"

    def find_pdf_link_from_soups(self, keys: List[Tuple[str, str]], title: str) -> Optional[str]:
        """Find a possible pdf link from soups for a given title and list of keys

        The keys correspond to the (venue, year) combination.
        The match is found by a greedy regexp match with first three tokens
        split on " ". The match with the longest span is returned.

        Args:
            keys: A list of (venue, year) tuples
            title: The title to match

        """
        links = []
        for k in keys:
            links.extend(self.soups[k].find_all("a"))
        if links:
            regexp = ".*" + ".*".join([*filter(None, title.split(" "))][:3]) + ".*\\.pdf$"
            matches = [*filter(None, map(lambda x: re.match(regexp, x["href"], flags=re.IGNORECASE)
                                         if "href" in x.attrs else None, links))]
            return self.best_match(title, matches)
        else:
            return None

    def find_pdf_link(self, keys: List[Tuple[str, str]], title: str) -> Optional[str]:
        """Find link from a list of pdf links for given keys and title.

        Similar to :meth:`find_pdf_link_from_soups` but it searches in already filtered
        pdf links.

        Args:
            keys: A list of (venue, year) tuples
            title: The title to match


        """
        links = []
        for k in keys:
            links.extend(self.cvf_pdf_links[k])
        if links:
            regexp = ".*" + ".*".join([*filter(None, title.split(" "))][:3]) + ".*\\.pdf$"
            matches = [*filter(None, map(lambda x: re.match(regexp, x, flags=re.IGNORECASE),
                                         links))]
            return self.best_match(title, matches)
        else:
            return None

    def maybe_download_cvf_day_pages(self, response: requests.Response,
                                     venue: str, year: str):
        """Maybe download the pages for each day of the conference

        The CVF links pages are sometimes split into days. We download all of the
        day pages and concatenate them into one big html for easy parsing.

        Args:
            response: An instance of :class:`requests.Response`
            venue: The venue
            year: The year


        """
        soup = BeautifulSoup(response.content, features="lxml")
        links = soup.find_all("a")
        regexp = f"(/)?({venue.upper()}{year}(.py)?).+"
        last_link_attrs = links[-1].attrs
        venue_match = re.match(regexp, last_link_attrs['href'])
        venue_group = venue_match and venue_match.group(2)
        if "href" in last_link_attrs and venue_group:
            day_links = [*filter(lambda x: re.match(r"Day [0-9]+?: ([0-9-+])", x.text),
                                 soup.find_all("a"))]
            content = []
            for i, dl in enumerate(day_links):
                maybe_matches = re.match(r"Day [0-9]+?: ([0-9-]+)", dl.text)
                if maybe_matches:
                    day = maybe_matches.groups()[0]
                else:
                    raise AttributeError(f"Could not find day {dl.text} in day links {day_links}")
                d_url = f"{self.cvf_url_root}/{venue_group}?day={day}"
                resp = self._get(d_url)
                if not resp.ok:
                    err = f"Status code {response.status_code} for {d_url}"
                    raise requests.HTTPError(err)
                content.append(resp.content)
                self.logger.debug(f"Fetched page {i+1} for {venue.upper()}{year} and {day}")
            soup_content = BeautifulSoup("")
            for c in content:
                soup_content.extend(BeautifulSoup(c, features="lxml").html)
            return soup_content.decode()
        self.logger.debug(f"Fetched page for {venue.upper()}{year}")
        return response.content.decode()

    def download_cvf_page_and_update_soups(self, venue, year):
        """Download a CVF page and update soups

        If required, pages for each day of the conference are downloaded
        and concatenated.

        Args:
            venue: Venue of conference
            year: Year of conference

        """
        url = f"{self.cvf_url_root}/{venue.upper()}{year}"
        response = self._get(url)
        if response.ok:
            content = self.maybe_download_cvf_day_pages(response, venue, year)
        else:
            err = f"Status code {response.status_code} for {url}"
            raise requests.HTTPError(err)
        fname = self.files_dir.joinpath(f"{venue.upper()}{year}")
        with open(fname, "w") as f:
            f.write(content)
        with open(fname) as f:
            self.soups[(venue.lower(), year)] = BeautifulSoup(content, features="lxml")

    def save_cvf_pdf_links_and_update(self, venue: str, year: str, soup) -> None:
        """Save the pdf links from parsed html and update cvf_pdf_links

        Args:
            venue: Venue of conference
            year: Year of conference
            soup: Html parsed as :class:`BeautifulSoup`

        """
        links = soup.find_all("a")
        pdf_links = [x["href"] for x in links
                     if "href" in x.attrs and x["href"].endswith(".pdf")]
        fname = self.files_dir.joinpath(f"{venue.upper()}{year}_pdfs")
        with open(fname, "w") as f:
            f.write("\n".join(pdf_links))
        self.cvf_pdf_links[(venue, year)] = pdf_links
