# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import mimetypes
import os
import pathlib
import re
import time
import uuid
import json
import chromadb
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse
import sys
import bm25s

import pathvalidate
import requests
#from serpapi import GoogleSearch
import serpapi

from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

from smolagents import Tool

from .cookies import COOKIES
from .mdconvert import FileConversionException, MarkdownConverter, UnsupportedFormatException

EMBEDDING_MODEL='Alibaba-NLP/gte-Qwen2-1.5B-instruct'
CROSS_ENCODING_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class OxCodeRag:
    def __init__(self, host:str, port:int, collection_name:str, local_dir:str, device_name:str='cpu'):
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._local_dir = local_dir
        self._chroma_client = chromadb.HttpClient(host=self._host, port=self._port)
        self._collection = self._chroma_client.get_collection(name=collection_name)
        self._embedding_model_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
        self._embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True).to(device_name)
        self._device_name = device_name
        self._cross_encoder_model = CrossEncoder(CROSS_ENCODING_MODEL)
        with open(os.path.join(self._local_dir, 'repos_info.json'), 'r') as rfp:
            self._repositories = json.loads(rfp.read())
        self._repository_list = list(map(str, self._repositories.keys()))
        self._bm25s_corpus_tokens = bm25s.tokenize(self._repository_list, lower=True)
        self._bm25s_retriever = bm25s.BM25(corpus=self._repository_list)
        self._bm25s_index = self._bm25s_retriever.index(self._bm25s_corpus_tokens)

    def _get_embedding(self, user_query):
        text = f"Instruct: Find the most relevant code snippet for the following query.\nQuery: {user_query}"
        batch_dict = self._embedding_model_tokenizer([text],
                                                padding=True,
                                                truncation=True,
                                                return_tensors='pt').to(self._device_name)
        outputs = self._embedding_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[:2]

    def _dump_sorted_results(self, results):
        for ind in range(len(results)):
            print(f"cross_distance={results[ind][0]}, id={results[ind][1]}, metadata={results[ind][2]}")
            # print(f"documents=\n{results[ind][3]}")
            print("=========================================================================")

    def get_relevant_code_snippet(self, user_query:str, repositories:str=None):
        print(f"OxCodeRag.get_relevant_code_snippet: Entered. user_query={user_query}, repositories={repositories}")
        emb = self._get_embedding(user_query)[0].detach().to('cpu').numpy()
        results = self._collection.query([emb], n_results=200)

        if repositories:
            cross_encoder_input = []
            cross_encoder_ids = []
            cross_encoder_metadatas = []
            cross_encoder_docs = []
            for ind in range(len(results['ids'][0])):
                for rn in repositories:
                    if rn in results['metadatas'][0][ind]['git']:
                        print(f"get_relevant_code_snippet: Yes! metadatas={results['metadatas'][0][ind]}")
                        cross_encoder_input.append((user_query, results['documents'][0][ind]))
                        cross_encoder_ids.append(results['ids'][0][ind])
                        cross_encoder_metadatas.append(results['metadatas'][0][ind])
                        cross_encoder_docs.append(results['documents'][0][ind])
                        break
        else:
            cross_encoder_input = [(user_query, results['documents'][0][ind]) for ind in range(len(results['ids'][0]))]
            cross_encoder_ids = results['ids'][0]
            cross_encoder_metadatas = results['metadatas'][0]
            cross_encoder_docs = results['documents'][0]

        if cross_encoder_input:
            cross_scores = self._cross_encoder_model.predict(cross_encoder_input)
            new_results = []
            for ind in range(len(cross_encoder_input)):
                new_results.append((cross_scores[ind],
                                cross_encoder_ids[ind],
                                cross_encoder_metadatas[ind],
                                cross_encoder_docs[ind]))
            sorted_results = sorted(new_results, reverse=True)
            self._cached_sorted_results = sorted_results
            self._cached_results_current_index = 0
            self._cached_user_query = user_query
            rv = sorted_results[self._cached_results_current_index][3]
        else:
            self._cached_sorted_results = []
            self._cached_results_current_index = None
            self._cached_user_query = user_query
            rv = []
        print(f"OxCodeRag.get_relevant_code_snippet: user_query={user_query}. Returning {rv}")
        return rv

    def get_filename_of_code_snippet(self, user_query:str):
        print(f"OxCodeRag.get_filename_of_code_snippet: Entered. user_query={user_query}")
        if self._cached_sorted_results and self._cached_user_query and user_query == self._cached_user_query:
            rv = self._cached_sorted_results[self._cached_results_current_index][2]['fn']
            print(f"OxCodeRag.get_filename_of_code_snippet: user_query={user_query}. Return {rv}")
            return rv
        else:
            print(f"OxCodeRag.get_filename_of_code_snippet: user_query={user_query}. Return empty string")
            return ""

    def list_repositories(self, user_query:str):
        print(f"OxCodeRag.list_repositories: Entered. user_query={user_query}")
        query_tokens = bm25s.tokenize(user_query, lower=True)
        k = min(5, len(self._repository_list))
        docs, scores = self._bm25s_retriever.retrieve(query_tokens, k=k)
        print(f"OxCodeRag.list_repositories: Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")
        return docs[0, 0]

class GetRelevantCodeSnippet(Tool):
    name = "get_relevant_code_snippet"
    description = "Given a user query and a list of company confidential git repositories, search each company confidential git repository mentioned and return the most relevant code snippet"
    inputs = {"user_query": {"type": "string", "description": "The user's query"},
            "repositories": {"type": "array", "items": {"type": "string"},
                    "description": "An array of strings, each string being the name of a repository to search for relevant code snippets"}}
    output_type = "string"

    def __init__(self, oxcoderag):
        super().__init__()
        self._oxcoderag = oxcoderag

    def forward(self, user_query: str, repositories: List[str]) -> str:
        return self._oxcoderag.get_relevant_code_snippet(user_query, repositories)

class GetFilenameOfCodeSnippet(Tool):
    name = "get_filename_of_code_snippet"
    description = "Return the name of the file containing the code snippet returned by GetRelevantCodeSnippet"
    inputs = {"user_query": {"type": "string", "description": "The user's query"}}
    output_type = "string"

    def __init__(self, oxcoderag):
        super().__init__()
        self._oxcoderag = oxcoderag

    def forward(self, user_query: str) -> str:
        return self._oxcoderag.get_filename_of_code_snippet(user_query)

class ListRepositories(Tool):
    name = "list_repositories"
    description = "Returns a comma separated list of company confidential git repositories known to this agent"
    inputs = {"user_query": {"type": "string", "description": "The user's query"}}
    output_type = "string"

    def __init__(self, oxcoderag):
        super().__init__()
        self._oxcoderag = oxcoderag

    def forward(self, user_query: str) -> str:
        return self._oxcoderag.list_repositories(user_query)


class SimpleTextBrowser:
    """(In preview) An extremely simple text-based web browser comparable to Lynx. Suitable for Agentic use."""

    def __init__(
        self,
        start_page: Optional[str] = None,
        viewport_size: Optional[int] = 1024 * 8,
        downloads_folder: Optional[Union[str, None]] = None,
        serpapi_key: Optional[Union[str, None]] = None,
        request_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    ):
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder
        self.history: List[Tuple[str, float]] = list()
        self.page_title: Optional[str] = None
        self.viewport_current_page = 0
        self.viewport_pages: List[Tuple[int, int]] = list()
        self.set_address(self.start_page)
        self.serpapi_key = serpapi_key
        self.request_kwargs = request_kwargs
        self.request_kwargs["cookies"] = COOKIES
        self._mdconvert = MarkdownConverter()
        self._page_content: str = ""

        self._find_on_page_query: Union[str, None] = None
        self._find_on_page_last_result: Union[int, None] = None  # Location of the last result

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1][0]

    def set_address(self, uri_or_path: str, filter_year: Optional[int] = None) -> None:
        # TODO: Handle anchors
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        elif uri_or_path.startswith("google:"):
            self._serpapi_search(uri_or_path[len("google:") :].strip(), filter_year=filter_year)
        else:
            if (
                not uri_or_path.startswith("http:")
                and not uri_or_path.startswith("https:")
                and not uri_or_path.startswith("file:")
            ):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # Update the address with the fully-qualified path
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0
        self.find_on_page_query = None
        self.find_on_page_viewport = None

    @property
    def viewport(self) -> str:
        """Return the content of the current viewport."""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]

    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """Sets the text content of the current page."""
        self._page_content = content
        self._split_pages()
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def find_on_page(self, query: str) -> Union[str, None]:
        """Searches for the query from the current viewport forward, looping back to the start if necessary."""

        # Did we get here via a previous find_on_page search with the same query?
        # If so, map to find_next
        if query == self._find_on_page_query and self.viewport_current_page == self._find_on_page_last_result:
            return self.find_next()

        # Ok it's a new search start from the current viewport
        self._find_on_page_query = query
        viewport_match = self._find_next_viewport(query, self.viewport_current_page)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def find_next(self) -> Union[str, None]:
        """Scroll to the next viewport that matches the query"""

        if self._find_on_page_query is None:
            return None

        starting_viewport = self._find_on_page_last_result
        if starting_viewport is None:
            starting_viewport = 0
        else:
            starting_viewport += 1
            if starting_viewport >= len(self.viewport_pages):
                starting_viewport = 0

        viewport_match = self._find_next_viewport(self._find_on_page_query, starting_viewport)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def _find_next_viewport(self, query: str, starting_viewport: int) -> Union[int, None]:
        """Search for matches between the starting viewport looping when reaching the end."""

        if query is None:
            return None

        # Normalize the query, and convert to a regular expression
        nquery = re.sub(r"\*", "__STAR__", query)
        nquery = " " + (" ".join(re.split(r"\W+", nquery))).strip() + " "
        nquery = nquery.replace(" __STAR__ ", "__STAR__ ")  # Merge isolated stars with prior word
        nquery = nquery.replace("__STAR__", ".*").lower()

        if nquery.strip() == "":
            return None

        idxs = list()
        idxs.extend(range(starting_viewport, len(self.viewport_pages)))
        idxs.extend(range(0, starting_viewport))

        for i in idxs:
            bounds = self.viewport_pages[i]
            content = self.page_content[bounds[0] : bounds[1]]

            # TODO: Remove markdown links and images
            ncontent = " " + (" ".join(re.split(r"\W+", content))).strip().lower() + " "
            if re.search(nquery, ncontent):
                return i

        return None

    def visit_page(self, path_or_uri: str, filter_year: Optional[int] = None) -> str:
        """Update the address, visit the page, and return the content of the viewport."""
        self.set_address(path_or_uri, filter_year=filter_year)
        return self.viewport

    def _split_pages(self) -> None:
        # Do not split search results
        if self.address.startswith("google:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx

    def _serpapi_search(self, query: str, filter_year: Optional[int] = None) -> None:
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        # search = GoogleSearch(params)
        search = serpapi.search(params)
        results = search.get_dict()
        self.page_title = f"{query} - Search"
        if "organic_results" not in results.keys():
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        if len(results["organic_results"]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            self._set_page_content(
                f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."
            )
            return

        def _prev_visit(url):
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == url:
                    return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
            return ""

        web_snippets: List[str] = list()
        idx = 0
        if "organic_results" in results:
            for page in results["organic_results"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{_prev_visit(page['link'])}{snippet}"

                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

        content = (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

        self._set_page_content(content)

    def _fetch_page(self, url: str) -> None:
        download_path = ""
        try:
            if url.startswith("file://"):
                download_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                res = self._mdconvert.convert_local(download_path)
                self.page_title = res.title
                self._set_page_content(res.text_content)
            else:
                # Prepare the request parameters
                request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
                request_kwargs["stream"] = True

                # Send a HTTP request to the URL
                response = requests.get(url, **request_kwargs)
                response.raise_for_status()

                # If the HTTP request was successful
                content_type = response.headers.get("content-type", "")

                # Text or HTML
                if "text/" in content_type.lower():
                    res = self._mdconvert.convert_response(response)
                    self.page_title = res.title
                    self._set_page_content(res.text_content)
                # A download
                else:
                    # Try producing a safe filename
                    fname = None
                    download_path = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                        suffix = 0
                        while os.path.exists(download_path) and suffix < 1000:
                            suffix += 1
                            base, ext = os.path.splitext(fname)
                            new_fname = f"{base}__{suffix}{ext}"
                            download_path = os.path.abspath(os.path.join(self.downloads_folder, new_fname))

                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                    # Open a file for writing
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Render it
                    local_uri = pathlib.Path(download_path).as_uri()
                    self.set_address(local_uri)

        except UnsupportedFormatException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileConversionException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileNotFoundError:
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
        except requests.exceptions.RequestException as request_exception:
            try:
                self.page_title = f"Error {response.status_code}"

                # If the error was rendered in HTML we might as well render it
                content_type = response.headers.get("content-type", "")
                if content_type is not None and "text/html" in content_type.lower():
                    res = self._mdconvert.convert(response)
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{res.text_content}")
                else:
                    text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        text += chunk
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{text}")
            except NameError:
                self.page_title = "Error"
                self._set_page_content(f"## Error\n\n{str(request_exception)}")

    def _state(self) -> Tuple[str, str]:
        header = f"Address: {self.address}\n"
        if self.page_title is not None:
            header += f"Title: {self.page_title}\n"

        current_page = self.viewport_current_page
        total_pages = len(self.viewport_pages)

        address = self.address
        for i in range(len(self.history) - 2, -1, -1):  # Start from the second last
            if self.history[i][0] == address:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                break

        header += f"Viewport position: Showing page {current_page + 1} of {total_pages}.\n"
        return (header, self.viewport)


class SearchInformationTool(Tool):
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {"query": {"type": "string", "description": "The web search query to perform."}}
    inputs["filter_year"] = {
        "type": "string",
        "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
        "nullable": True,
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        self.browser.visit_page(f"google: {query}", filter_year=filter_year)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class VisitTool(Tool):
    name = "visit_page"
    description = "Visit a webpage at a given URL and return its text. Given a url to a YouTube video, this returns the transcript."
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the webpage to visit."}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        self.browser.visit_page(url)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class DownloadTool(Tool):
    name = "download_file"
    description = """
Download a file at a given URL. The file should be of this format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".png", ".docx"]
After using this tool, for further inspection of this page you should return the download path to your manager via final_answer, and they will be able to inspect it.
DO NOT use this tool for .pdf or .txt or .htm files: for these types of files use visit_page with the file url instead."""
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the file to be downloaded."}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        if "arxiv" in url:
            url = url.replace("abs", "pdf")
        response = requests.get(url)
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type)
        if extension and isinstance(extension, str):
            new_path = f"./downloads/file{extension}"
        else:
            new_path = "./downloads/file.object"

        with open(new_path, "wb") as f:
            f.write(response.content)

        if "pdf" in extension or "txt" in extension or "htm" in extension:
            raise Exception("Do not use this tool for pdf or txt or html files: use visit_page instead.")

        return f"File was downloaded and saved under path {new_path}."


class ArchiveSearchTool(Tool):
    name = "find_archived_url"
    description = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."
    inputs = {
        "url": {"type": "string", "description": "The url you need the archive for."},
        "date": {
            "type": "string",
            "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'.",
        },
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url, date) -> str:
        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"
        response = requests.get(archive_url).json()
        response_notimestamp = requests.get(no_timestamp_url).json()
        if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
            closest = response["archived_snapshots"]["closest"]
            print("Archive found!", closest)

        elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp["archived_snapshots"]:
            closest = response_notimestamp["archived_snapshots"]["closest"]
            print("Archive found!", closest)
        else:
            raise Exception(f"Your {url=} was not archived on Wayback Machine, try a different url.")
        target_url = closest["url"]
        self.browser.visit_page(target_url)
        header, content = self.browser._state()
        return (
            f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
            + header.strip()
            + "\n=======================\n"
            + content
        )


class PageUpTool(Tool):
    name = "page_up"
    description = "Scroll the viewport UP one page-length in the current webpage and return the new viewport content."
    inputs = {}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        self.browser.page_up()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class PageDownTool(Tool):
    name = "page_down"
    description = (
        "Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        self.browser.page_down()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class FinderTool(Tool):
    name = "find_on_page_ctrl_f"
    description = "Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F."
    inputs = {
        "search_string": {
            "type": "string",
            "description": "The string to search for on the page. This search string supports wildcards like '*'",
        }
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, search_string: str) -> str:
        find_result = self.browser.find_on_page(search_string)
        header, content = self.browser._state()

        if find_result is None:
            return (
                header.strip()
                + f"\n=======================\nThe search string '{search_string}' was not found on this page."
            )
        else:
            return header.strip() + "\n=======================\n" + content


class FindNextTool(Tool):
    name = "find_next"
    description = "Scroll the viewport to next occurrence of the search string. This is equivalent to finding the next match in a Ctrl+F search."
    inputs = {}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        find_result = self.browser.find_next()
        header, content = self.browser._state()

        if find_result is None:
            return header.strip() + "\n=======================\nThe search string was not found on this page."
        else:
            return header.strip() + "\n=======================\n" + content


def main():
    if len(sys.argv) != 6:
        print(f"Usage: ox_coderag host port collection_name local_dir user_query")
        os._exit(255)
    #ox_code_rag = OxCodeRag(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], device_name='cuda')
    print(f">>>>>>>>>>>>>>>>>>>>>>>> Test 1")
    ox_code_rag = OxCodeRag(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
    reponame = ox_code_rag.list_repositories(sys.argv[5])
    print(f"list_repositories={reponame}")
    result = ox_code_rag.get_relevant_code_snippet(sys.argv[5])
    print(f"get_relevant_code_snippet={result}")
    result = ox_code_rag.get_filename_of_code_snippet(sys.argv[5])
    print(f"get_filename_of_code_snippet={result}")

    print(f">>>>>>>>>>>>>>>>>>>>>>>> Test 2")
    ox_code_rag = OxCodeRag(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
    reponame = ox_code_rag.list_repositories(sys.argv[5])
    print(f"list_repositories={reponame}")
    result = ox_code_rag.get_relevant_code_snippet(sys.argv[5], [reponame])
    print(f"get_relevant_code_snippet with reponame {reponame} ={result}")
    result = ox_code_rag.get_filename_of_code_snippet(sys.argv[5])
    print(f"get_filename_of_code_snippet={result}")

if __name__ == "__main__":
    main()
