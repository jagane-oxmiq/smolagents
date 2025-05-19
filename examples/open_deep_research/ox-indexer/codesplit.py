"""Code Splitter.

Loosely based on:
https://github.com/definitive-io/code-indexer-loop.git

Implementation amalgamated from:
https://docs.sweep.dev/blogs/chunking-improvements
https://docs.sweep.dev/blogs/chunking-2m-files
https://github.com/jerryjliu/llama_index/pull/7100

"""

import os
import sys
import json
import re
import base64
import pickle
import datetime
import time
import atexit
import traceback
from dataclasses import dataclass
from typing import List, Optional, Union
import requests

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from tree_sitter import Node
from tree_sitter_language_pack import get_language, get_parser

import chromadb

DEFAULT_EMBEDDING_MODEL='Alibaba-NLP/gte-Qwen2-1.5B-instruct'
DEFAULT_SUMMARIZER_MODEL='Qwen/Qwen2.5-Coder-7B-Instruct-AWQ'

repos_info_file = None
repos_info = {}

mtimes_file = None
files_info = {}

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class MaxChunkLengthExceededError(Exception):
    pass


@dataclass
class Span:
    # Represents a slice of a string
    start: int = 0
    end: int = 0

    def __post_init__(self):
        # If end is None, set it to start
        if self.end is None:
            self.end = self.start

    def extract(self, s: bytes) -> bytes:
        # Grab the corresponding substring of string s by bytes
        return s[self.start : self.end]

    def extract_lines(self, s: str) -> str:
        lines = s.split("\n")
        selected_lines = lines[self.start : self.end]
        joined = "\n".join(selected_lines)
        # if selection doesn't extend to the last line, add the missing newline
        if self.end < len(lines):
            joined += "\n"
        return joined

    def __add__(self, other: Union["Span", int]) -> "Span":
        # e.g. Span(1, 2) + Span(2, 4) = Span(1, 4) (concatenation)
        # There are no safety checks: Span(a, b) + Span(c, d) = Span(a, d)
        # and there are no requirements for b = c.
        if isinstance(other, int):
            return Span(self.start + other, self.end + other)
        elif isinstance(other, Span):
            return Span(self.start, other.end)
        else:
            raise NotImplementedError()

    def __len__(self) -> int:
        # i.e. Span(a, b) = b - a
        return self.end - self.start


class TokenCounter:
    model: str
    initialized_models = {}

    def __init__(self, model: str):
        self.model = model

    def count(self, text: str):
        if self.model not in self.initialized_models:
            try:
                self.initialized_models[self.model] = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            except KeyError:
                raise
        batch_dict = self.initialized_models[self.model]([text],
                                                    padding=True,
                                                    truncation=True,
                                                    return_tensors='pt')
        return batch_dict['input_ids'].shape[1]

    def count_chunk(self, chunk: Span, source_code: bytes):
        return self.count(chunk.extract(source_code).decode("utf-8"))

class TokenEmbedder:
    embedding_model: str
    initialized_tokenizers = {}
    initialized_models = {}

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL, device:str = 'cuda'):
        self.embedding_model = embedding_model
        self._device = device

    def embed(self, text: str):
        if self.embedding_model not in self.initialized_models:
            try:
                self.initialized_tokenizers[self.embedding_model] = AutoTokenizer.from_pretrained(self.embedding_model,
                                                                                    trust_remote_code=True)
                self.initialized_models[self.embedding_model] = AutoModel.from_pretrained(self.embedding_model,
                                                                                    trust_remote_code=True).to(self._device)
            except KeyError:
                raise
        batch_dict = self.initialized_tokenizers[self.embedding_model]([text],
                                                    padding=True,
                                                    truncation=True,
                                                    return_tensors='pt').to(self._device)
        outputs = self.initialized_models[self.embedding_model](**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[:2]

class CodeSplitter:
    """Split code using a AST parser."""

    language: str
    target_chunk_tokens: int
    max_chunk_tokens: int
    enforce_max_chunk_tokens: bool
    coalesce: int
    token_counter: TokenCounter

    def __init__(
        self,
        language: str,
        target_chunk_tokens: int,
        max_chunk_tokens: int,
        enforce_max_chunk_tokens: bool,
        coalesce: int,
        model: str
    ):
        self.token_counter = TokenCounter(model=model)
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.enforce_max_chunk_tokens = enforce_max_chunk_tokens
        self.language = language
        self.coalesce = coalesce

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "CodeSplitter"

    def chunk_tree(
        self,
        tree,
        source_code: bytes,
    ) -> list[Span]:
        # 1. Recursively form chunks
        def chunk_node(node: Node) -> list[Span]:
            chunks: list[Span] = []
            current_chunk: Span = Span(node.start_byte, node.start_byte)
            node_children = node.children
            for child in node_children:
                child_token_len = self.token_counter.count_chunk(Span(child.start_byte, child.end_byte), source_code)
                child_and_current_token_len = self.token_counter.count_chunk(
                    Span(child.start_byte, child.end_byte), source_code
                ) + self.token_counter.count_chunk(current_chunk, source_code)

                if child_token_len > self.target_chunk_tokens:
                    if child_token_len > self.max_chunk_tokens and self.enforce_max_chunk_tokens:
                        raise MaxChunkLengthExceededError(
                            f"Chunk token length {child_token_len} exceeds maximum {self.max_chunk_tokens}."
                        )

                    chunks.append(current_chunk)
                    current_chunk = Span(child.end_byte, child.end_byte)
                    chunks.extend(chunk_node(child))
                elif child_and_current_token_len > self.target_chunk_tokens:
                    if child_and_current_token_len > self.max_chunk_tokens and self.enforce_max_chunk_tokens:
                        raise MaxChunkLengthExceededError(
                            f"Chunk token length {child_and_current_token_len}"
                            f" exceeds maximum {self.max_chunk_tokens}."
                        )
                    chunks.append(current_chunk)
                    current_chunk = Span(child.start_byte, child.end_byte)
                else:
                    current_chunk += Span(child.start_byte, child.end_byte)

            final_chunk_token_len = self.token_counter.count_chunk(current_chunk, source_code)
            if final_chunk_token_len > self.max_chunk_tokens and self.enforce_max_chunk_tokens:
                raise MaxChunkLengthExceededError(
                    f"Chunk token length {final_chunk_token_len} exceeds maximum {self.max_chunk_tokens}."
                )
            chunks.append(current_chunk)
            return chunks

        chunks = chunk_node(tree.root_node)

        # Filter empty chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 0]

        # Early return if there is no chunk
        if len(chunks) == 0:
            return []
        # Early return if there is only one chunk
        if len(chunks) < 2:
            return [Span(0, len(chunks[0]))]

        # Filling in the gaps
        # by aligning end of one chunk with start of next
        chunks[0].start = 0
        for prev, curr in zip(chunks[:-1], chunks[1:]):
            prev.end = curr.start
        curr.end = len(source_code)

        # Combining small chunks with bigger ones
        new_chunks = []
        aggregated_chunk = Span(0, 0)
        aggregated_chunk_token_len = 0
        for chunk in chunks:
            # Check if the combined chunk exceeds target_chunk_tokens
            # Note, at this point no chunk exceeds max_chunk_tokens
            # if max_chunk_tokens is enforced.
            chunk_token_len = self.token_counter.count_chunk(chunk, source_code)
            if chunk_token_len > self.target_chunk_tokens:
                new_chunks.append(aggregated_chunk)
                new_chunks.append(chunk)
                aggregated_chunk = Span(chunk.end, chunk.end)
                aggregated_chunk_token_len = 0
            elif aggregated_chunk_token_len + chunk_token_len > self.target_chunk_tokens:
                new_chunks.append(aggregated_chunk)
                aggregated_chunk = Span(chunk.start, chunk.end)
                aggregated_chunk_token_len = chunk_token_len
            else:
                # Combined chunk does not exceed target_chunk_tokens
                # so we add the current chunk to the aggregated_chunk.
                # Note, there is no need to check whether the combined chunk
                # exceeds max_chunk_tokens because we have already checked.
                aggregated_chunk += chunk
                aggregated_chunk_token_len += chunk_token_len
                if aggregated_chunk_token_len > self.coalesce:
                    new_chunks.append(aggregated_chunk)
                    aggregated_chunk = Span(chunk.end, chunk.end)
                    aggregated_chunk_token_len = 0

        if len(aggregated_chunk) > 0:
            new_chunks.append(aggregated_chunk)

        # Changing line numbers
        line_chunks = [
            Span(
                self.get_line_number(chunk.start, source_code),
                self.get_line_number(chunk.end, source_code),
            )
            for chunk in new_chunks
        ]

        # Eliminating empty chunks
        line_chunks = [chunk for chunk in line_chunks if len(chunk) > 0]
        return line_chunks

    def split_and_keep_newline(self, byte_str):
        return re.split(b"(?<=\n)", byte_str)

    def get_line_number(self, index: int, source_code: bytes) -> int:
        total_chars = 0
        for line_number, line in enumerate(self.split_and_keep_newline(source_code), start=1):
            total_chars += len(line)
            if total_chars > index:
                return line_number - 1
        return line_number

    def split_text(self, text: str) -> List[str]:
        """Split incoming code and return chunks using the AST."""
        try:
            parser = get_parser(self.language)
        except Exception as e:
            print(
                f"Could not get parser for language {self.language}. Check "
                "https://github.com/grantjenks/py-tree-sitter-languages#license "
                "for a list of valid languages."
            )
            raise e

        tree = parser.parse(text.encode("utf-8"))
        if not tree.root_node.children or tree.root_node.children[0].type != "ERROR":
            line_spans = self.chunk_tree(tree, text.encode("utf-8"))
            chunks = [line_span.extract_lines(text) for line_span in line_spans]
            return chunks
        else:
            raise ValueError(f"Could not parse code with language {self.language}.")

class LocalCodeSummarizer:
    def __init__(self, model_name=DEFAULT_SUMMARIZER_MODEL, device=None):
        """
        Initialize the LocalCodeSummarizer with the Qwen2.5-Coder model.
        
        Args:
            model_name (str): HuggingFace model identifier
            device (str, optional): Device to run the model on ('cuda', 'cpu', etc.)
                                   If None, will use CUDA if available.
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load tokenizer and model
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True
        )
        print("LocalCodeSummarizer: Model loaded successfully!")
        
    def summarize(self, code_snippet, max_new_tokens=256, temperature=0.2):
        """
        Summarize the given code snippet.
        
        Args:
            code_snippet (str): The code to summarize
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling (lower = more deterministic)
            
        Returns:
            str: The generated summary of the code
        """
        # Create prompt in the format expected by Qwen models
        prompt = f"""<|im_start|>system
You are a helpful AI assistant that summarizes code concisely.
<|im_end|>
<|im_start|>user
Please summarize this code in a few sentences:
{code_snippet}
<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode the generated summary
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        summary = generated_text.split("<|im_start|>assistant")[-1]
        
        # Remove any trailing tags if present
        if "<|im_end|>" in summary:
            summary = summary.split("<|im_end|>")[0]
        
        return summary.strip()
    
    def __call__(self, code_snippet, max_new_tokens=256, temperature=0.2):
        """
        Allow the class to be called directly as a function.
        
        Args:
            Same as summarize() method
            
        Returns:
            str: The generated summary
        """
        return self.summarize(code_snippet, max_new_tokens, temperature)

class OpenAICodeSummarizer:
    def __init__(self, api_endpoint, api_key=None, model_name="Qwen/Qwen2.5-Coder-32B-Instruct"):
        """
        Initialize the OpenAICodeSummarizer to use an OpenAI-compatible API endpoint.
        
        Args:
            api_endpoint (str): The URL of the API endpoint (e.g. 'https://api.example.com/v1/chat/completions')
            api_key (str, optional): API key for authentication
            model_name (str): Model name to use in API requests
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_name = model_name
        
        # Prepare headers for API requests
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key to headers if provided
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
            
        print(f"OpenAICodeSummarizer initialized to use {model_name} via API endpoint")
    
    def summarize(self, code_snippet, max_tokens=256, temperature=0.2):
        """
        Summarize the given code snippet by calling the API.
        
        Args:
            code_snippet (str): The code to summarize
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling (lower = more deterministic)
            
        Returns:
            str: The generated summary of the code
        """
        prompt = f"You are an excellent AI assistant that summarizes code concisely. Summarize the following code in a few sentences:\n\n```\n{code_snippet}\n```"
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            # Make the API request
            response = requests.post(
                self.api_endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30  # 30 second timeout
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the generated text from the response
            # Format follows OpenAI's API response structure
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    # Standard OpenAI format
                    return response_data["choices"][0]["message"]["content"].strip()
                elif "text" in response_data["choices"][0]:
                    # Alternative format sometimes used
                    return response_data["choices"][0]["text"].strip()
            
            # If we couldn't find the expected structure, return the raw response
            return f"Error: Unexpected response format: {response_data}"
            
        except requests.exceptions.RequestException as e:
            # Handle request errors
            return f"Error: API request failed: {str(e)}"
        except json.JSONDecodeError:
            # Handle JSON parsing errors
            return f"Error: Failed to parse API response as JSON"
        except Exception as e:
            # Handle any other errors
            return f"Error: {str(e)}"
    
    def __call__(self, code_snippet, max_tokens=256, temperature=0.2):
        """
        Allow the class to be called directly as a function.
        
        Args:
            Same as summarize() method
            
        Returns:
            str: The generated summary
        """
        return self.summarize(code_snippet, max_tokens, temperature)

def process_file(git_repo_name:str, filename:str, code_splitter, token_embedder, code_summarizer, chroma_collection):
    global mtimes_file, files_info
    print(f"process_file: Entered. filename={filename}")
    if '.github' in filename:
        print(f"process_file: Skipping filename={filename} since it has .github in path")
        return
    elif '.vscode' in filename:
        print(f"process_file: Skipping filename={filename} since it has .vscode in path")
        return
    elif '.cpmcache' in filename:
        print(f"process_file: Skipping filename={filename} since it has .cpmcache in path")
        return
    elif 'site-packages' in filename:
        print(f"process_file: Skipping filename={filename} since it has site-packages in path")
        return
    mtime = os.path.getmtime(filename)
    if filename in files_info:
        if mtime <= files_info[filename]['mtime']:
            return
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    split_text = code_splitter.split_text(content)
    res = {'filename': filename, 'mtime': mtime}

    splits = []
    embeddings = []
    metadatas = []
    ids = []
    for split_number in range(len(split_text)):
        one_split = split_text[split_number]
        if code_summarizer:
            summary = code_summarizer(one_split)
        else:
            summary = one_split
        with torch.no_grad():
            embedding = token_embedder.embed(summary)
        splits.append(one_split)
        embeddings.append(embedding[0].detach().to('cpu').numpy())
        if code_summarizer:
            metadatas.append({"git": git_repo_name, "fn": filename, "split": str(split_number), "summary": summary})
        else:
            metadatas.append({"git": git_repo_name, "fn": filename, "split": str(split_number)})
        ids.append(f"{split_number}@{filename}")
    chroma_collection.add(documents=splits, embeddings=embeddings, metadatas=metadatas, ids=ids)
    files_info[filename] = res

def guess_language_from_filename(filename):
    """
    Guess the programming language of a file based on its filename/extension.
    
    Args:
        filename (str): The name of the file
        
    Returns:
        str: The guessed programming language or "Unknown" if not recognized
    """
    # Extract the file extension
    _, extension = os.path.splitext(filename)
    extension = extension.lower()
    
    # Map of file extensions to programming languages
    language_map = {
        ".actionscript": "actionscript",
        ".ada": "ada",
        ".agda": "agda",
        ".arduino": "arduino",
        ".asm": "asm",
        ".astro": "astro",
        ".sh": "bash",
        ".beancount": "beancount",
        ".bibtex": "bibtex",
        ".bicep": "bicep",
        ".bitbake": "bitbake",
        ".c": "c",
        ".cairo": "cairo",
        ".capnp": "capnp",
        ".chatito": "chatito",
        ".clarity": "clarity",
        ".cj": "clojure",
        ".cmake": "cmake",
        ".comment": "comment",
        ".commonlisp": "commonlisp",
        ".cpon": "cpon",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".css": "css",
        ".csv": "csv",
        ".cuda": "cuda",
        ".dart": "dart",
        ".dockerfile": "dockerfile",
        ".doxygen": "doxygen",
        ".dtd": "dtd",
        ".elisp": "elisp",
        ".ex": "elixir",
        ".elm": "elm",
        ".embeddedtemplate": "embeddedtemplate",
        ".erlang": "erlang",
        ".fennel": "fennel",
        ".firrtl": "firrtl",
        ".fish": "fish",
        ".fortran": "fortran",
        ".func": "func",
        ".gdscript": "gdscript",
        ".gitattributes": "gitattributes",
        ".gitcommit": "gitcommit",
        ".gitignore": "gitignore",
        ".gleam": "gleam",
        ".glsl": "glsl",
        ".gn": "gn",
        ".go": "go",
        ".gomod": "gomod",
        ".gosum": "gosum",
        ".groovy": "groovy",
        ".gstlaunch": "gstlaunch",
        ".hack": "hack",
        ".hare": "hare",
        ".hs": "haskell",
        ".haxe": "haxe",
        ".hcl": "hcl",
        ".heex": "heex",
        ".hlsl": "hlsl",
        ".html": "html",
        ".hyprlang": "hyprlang",
        ".ispc": "ispc",
        ".janet": "janet",
        ".java": "java",
        ".js": "javascript",
        ".jsdoc": "jsdoc",
        ".json": "json",
        ".jsonnet": "jsonnet",
        ".jl": "julia",
        ".kconfig": "kconfig",
        ".kdl": "kdl",
        ".kt": "kotlin",
        ".latex": "latex",
        ".linkerscript": "linkerscript",
        ".llvm": "llvm",
        ".lua": "lua",
        ".luadoc": "luadoc",
        ".luap": "luap",
        ".luau": "luau",
        ".make": "make",
        ".markdown": "markdown",
        ".matlab": "matlab",
        ".mermaid": "mermaid",
        ".meson": "meson",
        ".ninja": "ninja",
        ".nix": "nix",
        ".nqc": "nqc",
        ".m": "objc",
        ".odin": "odin",
        ".org": "org",
        ".pascal": "pascal",
        ".pem": "pem",
        ".pl": "perl",
        ".pgn": "pgn",
        ".php": "php",
        ".po": "po",
        ".pony": "pony",
        ".ps1": "powershell",
        ".printf": "printf",
        ".prisma": "prisma",
        ".properties": "properties",
        ".proto": "proto",
        ".psv": "psv",
        ".puppet": "puppet",
        ".purescript": "purescript",
        ".pymanifest": "pymanifest",
        ".py": "python",
        ".qmldir": "qmldir",
        ".qmljs": "qmljs",
        ".query": "query",
        ".r": "r",
        ".racket": "racket",
        ".re2c": "re2c",
        ".readline": "readline",
        ".requirements": "requirements",
        ".ron": "ron",
        ".rst": "rst",
        ".ruby": "ruby",
        ".rust": "rust",
        ".scala": "scala",
        ".scheme": "scheme",
        ".scss": "scss",
        ".smali": "smali",
        ".smithy": "smithy",
        ".solidity": "solidity",
        ".sparql": "sparql",
        ".swift": "swift",
        ".sql": "sql",
        ".squirrel": "squirrel",
        ".starlark": "starlark",
        ".svelte": "svelte",
        ".tablegen": "tablegen",
        ".tcl": "tcl",
        ".terraform": "terraform",
        ".test": "test",
        ".thrift": "thrift",
        ".toml": "toml",
        ".tsv": "tsv",
        ".tsx": "tsx",
        ".twig": "twig",
        ".ts": "typescript",
        ".typst": "typst",
        ".udev": "udev",
        ".ungrammar": "ungrammar",
        ".uxntal": "uxntal",
        ".v": "v",
        ".verilog": "verilog",
        ".vhdl": "vhdl",
        ".vim": "vim",
        ".vue": "vue",
        ".wgsl": "wgsl",
        ".xcompose": "xcompose",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yuck": "yuck",
        ".zig": "zig",
        ".magik": "magik"
    }
    
    if extension in language_map:
        return language_map[extension]
    else:
        # Special cases for filenames without extensions or ambiguous extensions
        if extension == '':
            if filename == 'Makefile' or filename == 'makefile':
                return 'make'
            elif filename == 'Dockerfile':
                return 'dockerfile'
            elif filename == 'Gemfile':
                return 'ruby'
            elif filename in ['README', 'LICENSE', 'CONTRIBUTING']:
                return 'txt'
        elif extension == '.md' or extension == '.markdown':
            return 'markdown'
        elif extension == '.json':
            return 'json'
        elif extension == '.xml':
            return 'xml'
        elif extension == '.yml' or extension == '.yaml':
            return 'yaml'
        return None

def scan_source_tree(git_repo_name, root_dir, code_splitters, embedding_model_name, token_embedder, code_summarizer, chroma_collection):
    print(f"scan_source_tree: Entered. root_dir={root_dir}")
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'errors': 0,
        'by_extension': {}
    }

    print(f"Starting source tree scan at: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            _, ext = os.path.splitext(filename)
            
            stats['total_files'] += 1
            language = guess_language_from_filename(filename)
            if not language:
                print(f"Skipping unknown language file {filename}")
                continue
            if not language in code_splitters:
                try:
                    code_splitters[language] = CodeSplitter(language, 512, 32768, True, 1, embedding_model_name)
                except Exception as ex:
                    print(f"Caught {ex} while trying to create code splitter for {language}")
                    continue

            # Update extension stats
            if ext not in stats['by_extension']:
                stats['by_extension'][ext] = 0
            stats['by_extension'][ext] += 1
            
            # Process the source file
            try:
                process_file(git_repo_name, file_path, code_splitters[language], token_embedder, code_summarizer, chroma_collection)
                stats['processed_files'] += 1
            except LookupError as e:
                pass
            except Exception as e:
                traceback.print_exc()
                stats['errors'] += 1
    return stats

def exit_handler():
    print(f"Program exiting. Writing out {mtimes_file} and {repos_info_file}")
    with open(mtimes_file, 'w') as ofile:
        for ky, val in files_info.items():
            ofile.write(f"{json.dumps(val)}\n")
    with open(repos_info_file, 'w') as ofile:
        ofile.write(f"{json.dumps(repos_info)}")

def main():
    if len(sys.argv) != 11:
        print(f"Usage: python codesplit.py <repos_info.json> <files_info.json> <chromadb_host> <chromadb_port> <chromadb_collection> <git_repo_name> <dir_with_local_copy_of_git_repo> summarizer_model|summarizer_url embedding_model device")
        os._exit(255)

    global mtimes_file, files_info
    atexit.register(exit_handler)
    mtimes_file = sys.argv[2]
    try:
        with open(mtimes_file, 'r') as ifile:
            for line in ifile:
                finfo = json.loads(line)
                filename = finfo['filename']
                files_info[filename] = finfo
    except Exception as ex:
        print(f"Caught {ex} opening {mtimes_file}. Ignoring and proceeding")

    global repos_info_file, repos_info
    repos_info_file = sys.argv[1]
    try:
        with open(repos_info_file, 'r') as ifile:
            repos_info = json.loads(ifile.read())
    except Exception as ex:
        print(f"Caught {ex} opening {repos_info_file}. Ignoring and proceeding")
    repos_info[sys.argv[6]] = {'dir': sys.argv[7]}

    chroma_client = chromadb.HttpClient(host=sys.argv[3], port=int(sys.argv[4]))
    chroma_collection = chroma_client.get_or_create_collection(name=sys.argv[5])

    code_splitters = {}
    token_embedder: TokenEmbedder = TokenEmbedder(sys.argv[9], device=sys.argv[10])
    if sys.argv[8].startswith('dontsummarize'):
        code_summarizer = None
    elif sys.argv[8].startswith('http'):
        code_summarizer: OpenAICodeSummarizer = OpenAICodeSummarizer(sys.argv[8], model_name="Qwen2.5-Coder-32B-Instruct")
    else:
        code_summarizer: LocalCodeSummarizer = LocalCodeSummarizer(model_name=sys.argv[8], device=sys.argv[9])

    scan_source_tree(sys.argv[6], sys.argv[7], code_splitters, sys.argv[9], token_embedder, code_summarizer, chroma_collection)

if __name__ == "__main__":
    main()
