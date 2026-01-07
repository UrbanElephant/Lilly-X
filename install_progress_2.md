Defaulting to user installation because normal site-packages is not writeable
Collecting llama-index (from -r requirements.txt (line 1))
  Using cached llama_index-0.14.12-py3-none-any.whl.metadata (13 kB)
Collecting llama-index-vector-stores-qdrant (from -r requirements.txt (line 2))
  Using cached llama_index_vector_stores_qdrant-0.1.4-py3-none-any.whl.metadata (696 bytes)
Collecting llama-index-embeddings-huggingface (from -r requirements.txt (line 3))
  Using cached llama_index_embeddings_huggingface-0.6.1-py3-none-any.whl.metadata (458 bytes)
Requirement already satisfied: python-dotenv in /home/gerrit/.local/lib/python3.14/site-packages (from -r requirements.txt (line 4)) (1.2.1)
Collecting llama-index-cli<0.6,>=0.5.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_cli-0.5.3-py3-none-any.whl.metadata (1.4 kB)
Collecting llama-index-core<0.15.0,>=0.14.12 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_core-0.14.12-py3-none-any.whl.metadata (2.5 kB)
Collecting llama-index-embeddings-openai<0.6,>=0.5.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_embeddings_openai-0.5.1-py3-none-any.whl.metadata (400 bytes)
Collecting llama-index-indices-managed-llama-cloud>=0.4.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_indices_managed_llama_cloud-0.9.4-py3-none-any.whl.metadata (3.7 kB)
Collecting llama-index-llms-openai<0.7,>=0.6.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_llms_openai-0.6.12-py3-none-any.whl.metadata (3.0 kB)
Collecting llama-index-readers-file<0.6,>=0.5.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_readers_file-0.5.6-py3-none-any.whl.metadata (5.7 kB)
Collecting llama-index-readers-llama-parse>=0.4.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_readers_llama_parse-0.5.1-py3-none-any.whl.metadata (3.1 kB)
Collecting nltk>3.8.1 (from llama-index->-r requirements.txt (line 1))
  Using cached nltk-3.9.2-py3-none-any.whl.metadata (3.2 kB)
Requirement already satisfied: aiohttp<4,>=3.8.6 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (3.13.2)
Requirement already satisfied: aiosqlite in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (0.21.0)
Collecting banks<3,>=2.2.0 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached banks-2.2.0-py3-none-any.whl.metadata (12 kB)
Requirement already satisfied: dataclasses-json in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (0.6.7)
Requirement already satisfied: deprecated>=1.2.9.3 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (1.3.1)
Collecting dirtyjson<2,>=1.0.8 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached dirtyjson-1.0.8-py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: filetype<2,>=1.2.0 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (1.2.0)
Requirement already satisfied: fsspec>=2023.5.0 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (2025.10.0)
Requirement already satisfied: httpx in /usr/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (0.28.1)
Collecting llama-index-workflows!=2.9.0,<3,>=2 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached llama_index_workflows-2.11.6-py3-none-any.whl.metadata (4.7 kB)
Requirement already satisfied: nest-asyncio<2,>=1.5.8 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (1.6.0)
Collecting networkx>=3.0 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached networkx-3.6.1-py3-none-any.whl.metadata (6.8 kB)
Requirement already satisfied: numpy in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (2.3.5)
Requirement already satisfied: pillow>=9.0.0 in /usr/lib64/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (11.3.0)
Collecting platformdirs (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached platformdirs-4.5.1-py3-none-any.whl.metadata (12 kB)
Requirement already satisfied: pydantic>=2.8.0 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (2.12.4)
Requirement already satisfied: pyyaml>=6.0.1 in /usr/lib64/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (6.0.2)
Requirement already satisfied: requests>=2.31.0 in /usr/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (2.32.5)
Collecting setuptools>=80.9.0 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached setuptools-80.9.0-py3-none-any.whl.metadata (6.6 kB)
Requirement already satisfied: sqlalchemy>=1.4.49 in /home/gerrit/.local/lib/python3.14/site-packages (from sqlalchemy[asyncio]>=1.4.49->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (2.0.44)
Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.2.0 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (9.1.2)
Requirement already satisfied: tiktoken>=0.7.0 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (0.12.0)
Requirement already satisfied: tqdm<5,>=4.66.1 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (4.67.1)
Requirement already satisfied: typing-extensions>=4.5.0 in /usr/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (4.15.0)
Requirement already satisfied: typing-inspect>=0.8.0 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (0.9.0)
Requirement already satisfied: wrapt in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (1.17.3)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /home/gerrit/.local/lib/python3.14/site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (2.6.1)
Requirement already satisfied: aiosignal>=1.4.0 in /home/gerrit/.local/lib/python3.14/site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (1.4.0)
Requirement already satisfied: attrs>=17.3.0 in /usr/lib/python3.14/site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (25.4.0)
Requirement already satisfied: frozenlist>=1.1.1 in /home/gerrit/.local/lib/python3.14/site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /home/gerrit/.local/lib/python3.14/site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (6.7.0)
Requirement already satisfied: propcache>=0.2.0 in /home/gerrit/.local/lib/python3.14/site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (0.4.1)
Requirement already satisfied: yarl<2.0,>=1.17.0 in /home/gerrit/.local/lib/python3.14/site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (1.22.0)
Collecting griffe (from banks<3,>=2.2.0->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached griffe-1.15.0-py3-none-any.whl.metadata (5.2 kB)
Requirement already satisfied: jinja2 in /home/gerrit/.local/lib/python3.14/site-packages (from banks<3,>=2.2.0->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (3.1.6)
Requirement already satisfied: openai>=1.1.0 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (2.8.0)
Requirement already satisfied: beautifulsoup4<5,>=4.12.3 in /usr/lib/python3.14/site-packages (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (4.14.3)
Collecting defusedxml>=0.7.1 (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached defusedxml-0.7.1-py2.py3-none-any.whl.metadata (32 kB)
Requirement already satisfied: pandas<3,>=2.0.0 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (2.3.3)
Collecting pypdf<7,>=6.1.3 (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached pypdf-6.5.0-py3-none-any.whl.metadata (7.1 kB)
Collecting striprtf<0.0.27,>=0.0.26 (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached striprtf-0.0.26-py3-none-any.whl.metadata (2.1 kB)
Requirement already satisfied: soupsieve>=1.6.1 in /usr/lib/python3.14/site-packages (from beautifulsoup4<5,>=4.12.3->llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (2.8)
Collecting llama-index-instrumentation>=0.1.0 (from llama-index-workflows!=2.9.0,<3,>=2->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached llama_index_instrumentation-0.4.2-py3-none-any.whl.metadata (1.1 kB)
Requirement already satisfied: anyio<5,>=3.5.0 in /home/gerrit/.local/lib/python3.14/site-packages (from openai>=1.1.0->llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (4.12.0)
Requirement already satisfied: distro<2,>=1.7.0 in /usr/lib/python3.14/site-packages (from openai>=1.1.0->llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (1.9.0)
Requirement already satisfied: jiter<1,>=0.10.0 in /home/gerrit/.local/lib/python3.14/site-packages (from openai>=1.1.0->llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (0.12.0)
Requirement already satisfied: sniffio in /usr/lib/python3.14/site-packages (from openai>=1.1.0->llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (1.3.1)
Requirement already satisfied: idna>=2.8 in /usr/lib/python3.14/site-packages (from anyio<5,>=3.5.0->openai>=1.1.0->llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (3.10)
Requirement already satisfied: certifi in /usr/lib/python3.14/site-packages (from httpx->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (2025.7.9)
Requirement already satisfied: httpcore==1.* in /usr/lib/python3.14/site-packages (from httpx->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (1.0.9)
Requirement already satisfied: h11>=0.16 in /usr/lib/python3.14/site-packages (from httpcore==1.*->httpx->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (0.16.0)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/lib/python3.14/site-packages (from pandas<3,>=2.0.0->llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /home/gerrit/.local/lib/python3.14/site-packages (from pandas<3,>=2.0.0->llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /home/gerrit/.local/lib/python3.14/site-packages (from pandas<3,>=2.0.0->llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1)) (2025.2)
Requirement already satisfied: annotated-types>=0.6.0 in /home/gerrit/.local/lib/python3.14/site-packages (from pydantic>=2.8.0->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (0.7.0)
Requirement already satisfied: pydantic-core==2.41.5 in /home/gerrit/.local/lib/python3.14/site-packages (from pydantic>=2.8.0->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (2.41.5)
Requirement already satisfied: typing-inspection>=0.4.2 in /home/gerrit/.local/lib/python3.14/site-packages (from pydantic>=2.8.0->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1)) (0.4.2)
Requirement already satisfied: grpcio<2.0.0,>=1.60.0 in /home/gerrit/.local/lib/python3.14/site-packages (from llama-index-vector-stores-qdrant->-r requirements.txt (line 2)) (1.76.0)
INFO: pip is looking at multiple versions of llama-index-vector-stores-qdrant to determine which version is compatible with other requirements. This could take a while.
Collecting llama-index-vector-stores-qdrant (from -r requirements.txt (line 2))
  Using cached llama_index_vector_stores_qdrant-0.1.3-py3-none-any.whl.metadata (749 bytes)
Collecting yarl<2.0,>=1.17.0 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached yarl-1.22.0-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (75 kB)
Collecting typing-extensions>=4.5.0 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting tqdm<5,>=4.66.1 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
INFO: pip is still looking at multiple versions of llama-index-vector-stores-qdrant to determine which version is compatible with other requirements. This could take a while.
Collecting tenacity!=8.4.0,<10.0.0,>=8.2.0 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached tenacity-9.1.2-py3-none-any.whl.metadata (1.2 kB)
Collecting pypdf<7,>=6.1.3 (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached pypdf-6.4.2-py3-none-any.whl.metadata (7.1 kB)
Collecting pydantic-core==2.41.5 (from pydantic>=2.8.0->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached pydantic_core-2.41.5-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
Collecting pydantic>=2.8.0 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
Collecting pandas<3,>=2.0.0 (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached pandas-2.3.3-cp314-cp314-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (91 kB)
Collecting jiter<1,>=0.10.0 (from openai>=1.1.0->llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached jiter-0.12.0-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.2 kB)
Collecting httpcore==1.* (from httpx->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
Collecting httpx (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting distro<2,>=1.7.0 (from openai>=1.1.0->llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached distro-1.9.0-py3-none-any.whl.metadata (6.8 kB)
Collecting anyio<5,>=3.5.0 (from openai>=1.1.0->llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached anyio-4.12.0-py3-none-any.whl.metadata (4.3 kB)
Collecting openai>=1.1.0 (from llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached openai-2.14.0-py3-none-any.whl.metadata (29 kB)
Collecting nest-asyncio<2,>=1.5.8 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached nest_asyncio-1.6.0-py3-none-any.whl.metadata (2.8 kB)
Collecting multidict<7.0,>=4.5 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached multidict-6.7.0-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (5.3 kB)
Collecting llama-index-workflows!=2.9.0,<3,>=2 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached llama_index_workflows-2.11.5-py3-none-any.whl.metadata (4.7 kB)
Collecting beautifulsoup4<5,>=4.12.3 (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached beautifulsoup4-4.14.3-py3-none-any.whl.metadata (3.8 kB)
Collecting llama-index-readers-file<0.6,>=0.5.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_readers_file-0.5.5-py3-none-any.whl.metadata (5.7 kB)
Collecting pandas<2.3.0 (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached pandas-2.2.3-cp314-cp314-linux_x86_64.whl
Collecting llama-index-readers-file<0.6,>=0.5.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_readers_file-0.5.4-py3-none-any.whl.metadata (5.7 kB)
  Using cached llama_index_readers_file-0.5.3-py3-none-any.whl.metadata (5.7 kB)
INFO: pip is looking at multiple versions of llama-index-readers-file to determine which version is compatible with other requirements. This could take a while.
  Using cached llama_index_readers_file-0.5.2-py3-none-any.whl.metadata (5.7 kB)
  Using cached llama_index_readers_file-0.5.1-py3-none-any.whl.metadata (5.7 kB)
  Using cached llama_index_readers_file-0.5.0-py3-none-any.whl.metadata (5.3 kB)
Collecting llama-index-llms-openai<0.7,>=0.6.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_llms_openai-0.6.11-py3-none-any.whl.metadata (3.0 kB)
INFO: pip is still looking at multiple versions of llama-index-readers-file to determine which version is compatible with other requirements. This could take a while.
  Using cached llama_index_llms_openai-0.6.10-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.6.9-py3-none-any.whl.metadata (3.0 kB)
INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
  Using cached llama_index_llms_openai-0.6.8-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.6.7-py3-none-any.whl.metadata (3.0 kB)
Collecting openai>=1.1.0 (from llama-index-embeddings-openai<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached openai-1.109.1-py3-none-any.whl.metadata (29 kB)
  Using cached openai-1.109.0-py3-none-any.whl.metadata (29 kB)
Collecting llama-index-llms-openai<0.7,>=0.6.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_llms_openai-0.6.6-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.6.5-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.6.4-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.6.3-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.6.2-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.6.1-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.6.0-py3-none-any.whl.metadata (3.0 kB)
Collecting llama-index-embeddings-openai<0.6,>=0.5.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_embeddings_openai-0.5.0-py3-none-any.whl.metadata (400 bytes)
INFO: pip is looking at multiple versions of llama-index-embeddings-openai to determine which version is compatible with other requirements. This could take a while.
Collecting filetype<2,>=1.2.0 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached filetype-1.2.0-py2.py3-none-any.whl.metadata (6.5 kB)
Collecting aiohttp<4,>=3.8.6 (from llama-index-core<0.15.0,>=0.14.12->llama-index->-r requirements.txt (line 1))
  Using cached aiohttp-3.13.2-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (8.1 kB)
  Using cached aiohttp-3.13.1-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (8.1 kB)
  Using cached aiohttp-3.13.0-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (8.1 kB)
Collecting llama-index-cli<0.6,>=0.5.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_cli-0.5.2-py3-none-any.whl.metadata (1.4 kB)
  Using cached llama_index_cli-0.5.1-py3-none-any.whl.metadata (1.4 kB)
INFO: pip is looking at multiple versions of llama-index-cli to determine which version is compatible with other requirements. This could take a while.
  Using cached llama_index_cli-0.5.0-py3-none-any.whl.metadata (1.4 kB)
Collecting llama-index (from -r requirements.txt (line 1))
  Using cached llama_index-0.14.10-py3-none-any.whl.metadata (13 kB)
INFO: pip is still looking at multiple versions of llama-index-embeddings-openai to determine which version is compatible with other requirements. This could take a while.
  Using cached llama_index-0.14.9-py3-none-any.whl.metadata (13 kB)
INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
  Using cached llama_index-0.14.8-py3-none-any.whl.metadata (13 kB)
INFO: pip is still looking at multiple versions of llama-index-cli to determine which version is compatible with other requirements. This could take a while.
  Using cached llama_index-0.14.7-py3-none-any.whl.metadata (13 kB)
  Using cached llama_index-0.14.6-py3-none-any.whl.metadata (13 kB)
  Using cached llama_index-0.14.5-py3-none-any.whl.metadata (13 kB)
INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
  Using cached llama_index-0.14.4-py3-none-any.whl.metadata (13 kB)
  Using cached llama_index-0.14.3-py3-none-any.whl.metadata (13 kB)
Collecting llama-index-llms-openai<0.6,>=0.5.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_llms_openai-0.5.6-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.5.5-py3-none-any.whl.metadata (3.0 kB)
INFO: pip is looking at multiple versions of llama-index-llms-openai to determine which version is compatible with other requirements. This could take a while.
  Using cached llama_index_llms_openai-0.5.4-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.5.3-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.5.2-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.5.1-py3-none-any.whl.metadata (3.0 kB)
  Using cached llama_index_llms_openai-0.5.0-py3-none-any.whl.metadata (3.0 kB)
Collecting llama-index (from -r requirements.txt (line 1))
  Using cached llama_index-0.14.2-py3-none-any.whl.metadata (13 kB)
INFO: pip is still looking at multiple versions of llama-index-llms-openai to determine which version is compatible with other requirements. This could take a while.
  Using cached llama_index-0.14.1-py3-none-any.whl.metadata (13 kB)
INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
  Using cached llama_index-0.14.0-py3-none-any.whl.metadata (12 kB)
Collecting llama-index-core<0.15,>=0.13.6 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_core-0.13.6-py3-none-any.whl.metadata (2.5 kB)
Collecting llama-index-workflows<2,>=1.0.1 (from llama-index-core<0.15,>=0.13.6->llama-index->-r requirements.txt (line 1))
  Using cached llama_index_workflows-1.3.0-py3-none-any.whl.metadata (6.4 kB)
Collecting pypdf<6,>=5.1.0 (from llama-index-readers-file<0.6,>=0.5.0->llama-index->-r requirements.txt (line 1))
  Using cached pypdf-5.9.0-py3-none-any.whl.metadata (7.1 kB)
Collecting llama-index (from -r requirements.txt (line 1))
  Using cached llama_index-0.13.6-py3-none-any.whl.metadata (12 kB)
Collecting llama-index-workflows<2,>=1.0.1 (from llama-index-core<0.15,>=0.13.6->llama-index->-r requirements.txt (line 1))
  Using cached llama_index_workflows-1.2.0-py3-none-any.whl.metadata (6.2 kB)
Collecting llama-index (from -r requirements.txt (line 1))
  Using cached llama_index-0.13.5-py3-none-any.whl.metadata (12 kB)
  Using cached llama_index-0.13.4-py3-none-any.whl.metadata (12 kB)
  Using cached llama_index-0.13.3-py3-none-any.whl.metadata (12 kB)
  Using cached llama_index-0.13.2-py3-none-any.whl.metadata (12 kB)
  Using cached llama_index-0.13.1-py3-none-any.whl.metadata (12 kB)
  Using cached llama_index-0.13.0-py3-none-any.whl.metadata (12 kB)
  Using cached llama_index-0.12.52-py3-none-any.whl.metadata (12 kB)
Collecting llama-index-agent-openai<0.5,>=0.4.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_agent_openai-0.4.12-py3-none-any.whl.metadata (439 bytes)
Collecting llama-index-cli<0.5,>=0.4.2 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_cli-0.4.4-py3-none-any.whl.metadata (1.4 kB)
Collecting llama-index-core<0.13,>=0.12.52.post1 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_core-0.12.52.post1-py3-none-any.whl.metadata (2.5 kB)
Collecting llama-index-embeddings-openai<0.4,>=0.3.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_embeddings_openai-0.3.1-py3-none-any.whl.metadata (684 bytes)
Collecting llama-index-llms-openai<0.5,>=0.4.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_llms_openai-0.4.7-py3-none-any.whl.metadata (3.0 kB)
Collecting llama-index-multi-modal-llms-openai<0.6,>=0.5.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_multi_modal_llms_openai-0.5.3-py3-none-any.whl.metadata (441 bytes)
Collecting llama-index-program-openai<0.4,>=0.3.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_program_openai-0.3.2-py3-none-any.whl.metadata (473 bytes)
Collecting llama-index-question-gen-openai<0.4,>=0.3.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_question_gen_openai-0.3.1-py3-none-any.whl.metadata (492 bytes)
Collecting llama-index-readers-file<0.5,>=0.4.0 (from llama-index->-r requirements.txt (line 1))
  Using cached llama_index_readers_file-0.4.11-py3-none-any.whl.metadata (5.3 kB)
Collecting pypdf<6,>=5.1.0 (from llama-index-readers-file<0.5,>=0.4.0->llama-index->-r requirements.txt (line 1))
  Using cached pypdf-5.8.0-py3-none-any.whl.metadata (7.1 kB)
Collecting pandas<2.3.0 (from llama-index-readers-file<0.5,>=0.4.0->llama-index->-r requirements.txt (line 1))
  Using cached pandas-2.2.2.tar.gz (4.4 MB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Installing backend dependencies: started
  Installing backend dependencies: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'error'
  error: subprocess-exited-with-error
  
  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [166 lines of output]
      + meson setup /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914 /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/.mesonpy-5b26ks4c/build -Dbuildtype=release -Db_ndebug=if-release -Db_vscrt=md --vsenv --native-file=/tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/.mesonpy-5b26ks4c/build/meson-python-native-file.ini
      The Meson build system
      Version: 1.2.1
      Source dir: /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914
      Build dir: /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/.mesonpy-5b26ks4c/build
      Build type: native build
      Project name: pandas
      Project version: 2.2.2
      C compiler for the host machine: cc (gcc 15.2.1 "cc (GCC) 15.2.1 20251211 (Red Hat 15.2.1-5)")
      C linker for the host machine: cc ld.bfd 2.45.1-1
      C++ compiler for the host machine: c++ (gcc 15.2.1 "c++ (GCC) 15.2.1 20251211 (Red Hat 15.2.1-5)")
      C++ linker for the host machine: c++ ld.bfd 2.45.1-1
      Cython compiler for the host machine: cython (cython 3.0.5)
      Host machine cpu family: x86_64
      Host machine cpu: x86_64
      Program python found: YES (/usr/bin/python)
      Found pkg-config: /usr/bin/pkg-config (2.3.0)
      Run-time dependency python found: YES 3.14
      Build targets in project: 53
      
      pandas 2.2.2
      
        User defined options
          Native files: /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/.mesonpy-5b26ks4c/build/meson-python-native-file.ini
          buildtype   : release
          vsenv       : True
          b_ndebug    : if-release
          b_vscrt     : md
      
      Found ninja-1.13.0.git.kitware.jobserver-pipe-1 at /tmp/pip-build-env-wtfwz3bc/normal/bin/ninja
      
      Visual Studio environment is needed to run Ninja. It is recommended to use Meson wrapper:
      /tmp/pip-build-env-wtfwz3bc/overlay/bin/meson compile -C .
      + /tmp/pip-build-env-wtfwz3bc/normal/bin/ninja
      [1/151] Generating pandas/_libs/algos_common_helper_pxi with a custom command
      [2/151] Generating pandas/_libs/algos_take_helper_pxi with a custom command
      [3/151] Generating pandas/_libs/intervaltree_helper_pxi with a custom command
      [4/151] Generating pandas/_libs/hashtable_class_helper_pxi with a custom command
      [5/151] Generating pandas/_libs/hashtable_func_helper_pxi with a custom command
      [6/151] Generating pandas/_libs/khash_primitive_helper_pxi with a custom command
      [7/151] Generating pandas/_libs/index_class_helper_pxi with a custom command
      [8/151] Generating pandas/_libs/sparse_op_helper_pxi with a custom command
      [9/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/ops_dispatch.pyx
      [10/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/properties.pyx
      [11/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/byteswap.pyx
      [12/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/base.pyx
      [13/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/ccalendar.pyx
      [14/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/indexing.pyx
      [15/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/dtypes.pyx
      [16/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/testing.pyx
      [17/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/nattype.pyx
      warning: /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/nattype.pyx:79:0: Global name __nat_unpickle matched from within class scope in contradiction to to Python 'class private name' rules. This may change in a future release.
      warning: /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/nattype.pyx:79:0: Global name __nat_unpickle matched from within class scope in contradiction to to Python 'class private name' rules. This may change in a future release.
      [18/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/missing.pyx
      [19/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/np_datetime.pyx
      [20/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/arrays.pyx
      [21/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/sas.pyx
      [22/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/hashing.pyx
      [23/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/ops.pyx
      [24/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/timezones.pyx
      [25/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/fields.pyx
      [26/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/vectorized.pyx
      [27/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/tzconversion.pyx
      [28/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/reshape.pyx
      [29/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/conversion.pyx
      [30/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslib.pyx
      [31/151] Compiling C object pandas/_libs/tslibs/base.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_base.pyx.c.o
      [32/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/internals.pyx
      [33/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/parsing.pyx
      [34/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/strptime.pyx
      [35/151] Compiling C object pandas/_libs/tslibs/parsing.cpython-314-x86_64-linux-gnu.so.p/.._src_parser_tokenizer.c.o
      [36/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/timestamps.pyx
      [37/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/period.pyx
      [38/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/offsets.pyx
      [39/151] Compiling C object pandas/_libs/pandas_datetime.cpython-314-x86_64-linux-gnu.so.p/src_vendored_numpy_datetime_np_datetime.c.o
      [40/151] Compiling C object pandas/_libs/lib.cpython-314-x86_64-linux-gnu.so.p/src_parser_tokenizer.c.o
      [41/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/window/indexers.pyx
      [42/151] Compiling C object pandas/_libs/pandas_datetime.cpython-314-x86_64-linux-gnu.so.p/src_datetime_date_conversions.c.o
      [43/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/tslibs/timedeltas.pyx
      [44/151] Compiling C object pandas/_libs/pandas_datetime.cpython-314-x86_64-linux-gnu.so.p/src_vendored_numpy_datetime_np_datetime_strings.c.o
      [45/151] Compiling C object pandas/_libs/pandas_parser.cpython-314-x86_64-linux-gnu.so.p/src_parser_io.c.o
      [46/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/writers.pyx
      [47/151] Compiling C object pandas/_libs/pandas_datetime.cpython-314-x86_64-linux-gnu.so.p/src_datetime_pd_datetime.c.o
      [48/151] Compiling C object pandas/_libs/pandas_parser.cpython-314-x86_64-linux-gnu.so.p/src_parser_pd_parser.c.o
      [49/151] Compiling C object pandas/_libs/parsers.cpython-314-x86_64-linux-gnu.so.p/src_parser_io.c.o
      [50/151] Compiling C object pandas/_libs/indexing.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_indexing.pyx.c.o
      [51/151] Compiling C object pandas/_libs/tslibs/ccalendar.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_ccalendar.pyx.c.o
      [52/151] Compiling C object pandas/_libs/json.cpython-314-x86_64-linux-gnu.so.p/src_vendored_ujson_python_ujson.c.o
      [53/151] Compiling C object pandas/_libs/json.cpython-314-x86_64-linux-gnu.so.p/src_vendored_ujson_python_JSONtoObj.c.o
      [54/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/lib.pyx
      [55/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/index.pyx
      [56/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/parsers.pyx
      [57/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/window/aggregations.pyx
      [58/151] Compiling C object pandas/_libs/json.cpython-314-x86_64-linux-gnu.so.p/src_vendored_ujson_lib_ultrajsondec.c.o
      [59/151] Compiling C object pandas/_libs/pandas_parser.cpython-314-x86_64-linux-gnu.so.p/src_parser_tokenizer.c.o
      [60/151] Compiling C object pandas/_libs/json.cpython-314-x86_64-linux-gnu.so.p/src_vendored_ujson_lib_ultrajsonenc.c.o
      [61/151] Compiling C object pandas/_libs/parsers.cpython-314-x86_64-linux-gnu.so.p/src_parser_tokenizer.c.o
      [62/151] Compiling C object pandas/_libs/json.cpython-314-x86_64-linux-gnu.so.p/src_vendored_ujson_python_objToJSON.c.o
      [63/151] Compiling C object pandas/_libs/arrays.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_arrays.pyx.c.o
      [64/151] Compiling C object pandas/_libs/ops_dispatch.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_ops_dispatch.pyx.c.o
      [65/151] Compiling C object pandas/_libs/tslibs/np_datetime.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_np_datetime.pyx.c.o
      [66/151] Compiling C object pandas/_libs/byteswap.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_byteswap.pyx.c.o
      [67/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/sparse.pyx
      [68/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/interval.pyx
      [69/151] Compiling C object pandas/_libs/properties.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_properties.pyx.c.o
      [70/151] Compiling C++ object pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_window_aggregations.pyx.cpp.o
      FAILED: [code=1] pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_window_aggregations.pyx.cpp.o
      c++ -Ipandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p -Ipandas/_libs/window -I../../pandas/_libs/window -I../../../../pip-build-env-wtfwz3bc/overlay/lib64/python3.14/site-packages/numpy/_core/include -I../../pandas/_libs/include -I/usr/include/python3.14 -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-color=always -DNDEBUG -D_FILE_OFFSET_BITS=64 -w -O3 -DNPY_NO_DEPRECATED_API=0 -DNPY_TARGET_VERSION=NPY_1_21_API_VERSION -fPIC -MD -MQ pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_window_aggregations.pyx.cpp.o -MF pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_window_aggregations.pyx.cpp.o.d -o pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_window_aggregations.pyx.cpp.o -c pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/window/aggregations.pyx.cpp
      pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/window/aggregations.pyx.cpp:422:31: Fehler: Standardattribute inmitten von Deklarationssymbolen
        422 |         #define CYTHON_UNUSED [[maybe_unused]]
            |                               ^
      pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/window/aggregations.pyx.cpp:2260:12: Anmerkung: bei Substitution des Makros »CYTHON_UNUSED«
       2260 |     static CYTHON_UNUSED PyObject *__Pyx_KwargsAsDict_FASTCALL(PyObject *kwnames, PyObject *const *kwvalues);
            |            ^~~~~~~~~~~~~
      pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/window/aggregations.pyx.cpp:422:31: Anmerkung: Standardattribute müssen vor den Deklarationsspezifizierern stehen, um für die Deklaration zu gelten, oder ihnen folgen, um für den Typ zu gelten
        422 |         #define CYTHON_UNUSED [[maybe_unused]]
            |                               ^
      pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/window/aggregations.pyx.cpp:2260:12: Anmerkung: bei Substitution des Makros »CYTHON_UNUSED«
       2260 |     static CYTHON_UNUSED PyObject *__Pyx_KwargsAsDict_FASTCALL(PyObject *kwnames, PyObject *const *kwvalues);
            |            ^~~~~~~~~~~~~
      pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/window/aggregations.pyx.cpp:422:31: Fehler: Standardattribute inmitten von Deklarationssymbolen
        422 |         #define CYTHON_UNUSED [[maybe_unused]]
            |                               ^
      pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/window/aggregations.pyx.cpp:42115:8: Anmerkung: bei Substitution des Makros »CYTHON_UNUSED«
      42115 | static CYTHON_UNUSED PyObject *__Pyx_KwargsAsDict_FASTCALL(PyObject *kwnames, PyObject *const *kwvalues) {
            |        ^~~~~~~~~~~~~
      pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/window/aggregations.pyx.cpp:422:31: Anmerkung: Standardattribute müssen vor den Deklarationsspezifizierern stehen, um für die Deklaration zu gelten, oder ihnen folgen, um für den Typ zu gelten
        422 |         #define CYTHON_UNUSED [[maybe_unused]]
            |                               ^
      pandas/_libs/window/aggregations.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/window/aggregations.pyx.cpp:42115:8: Anmerkung: bei Substitution des Makros »CYTHON_UNUSED«
      42115 | static CYTHON_UNUSED PyObject *__Pyx_KwargsAsDict_FASTCALL(PyObject *kwnames, PyObject *const *kwvalues) {
            |        ^~~~~~~~~~~~~
      [71/151] Generating pandas/__init__.py with a custom command
      [72/151] Compiling C object pandas/_libs/tslibs/nattype.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_nattype.pyx.c.o
      [73/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/join.pyx
      [74/151] Compiling C object pandas/_libs/tslibs/dtypes.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_dtypes.pyx.c.o
      [75/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/hashtable.pyx
      [76/151] Compiling C object pandas/_libs/hashing.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_hashing.pyx.c.o
      [77/151] Compiling C object pandas/_libs/tslibs/vectorized.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_vectorized.pyx.c.o
      [78/151] Compiling C object pandas/_libs/missing.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_missing.pyx.c.o
      [79/151] Compiling C object pandas/_libs/testing.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_testing.pyx.c.o
      [80/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/algos.pyx
      [81/151] Compiling Cython source /tmp/pip-install-cchhepvn/pandas_742520255f5e46e58194cf9aa510c914/pandas/_libs/groupby.pyx
      [82/151] Compiling C object pandas/_libs/tslibs/timezones.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_timezones.pyx.c.o
      [83/151] Compiling C object pandas/_libs/window/indexers.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_window_indexers.pyx.c.o
      [84/151] Compiling C object pandas/_libs/tslibs/conversion.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_conversion.pyx.c.o
      [85/151] Compiling C object pandas/_libs/sas.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_sas.pyx.c.o
      [86/151] Compiling C object pandas/_libs/ops.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_ops.pyx.c.o
      [87/151] Compiling C object pandas/_libs/tslibs/tzconversion.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_tzconversion.pyx.c.o
      [88/151] Compiling C object pandas/_libs/tslibs/fields.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_fields.pyx.c.o
      [89/151] Compiling C object pandas/_libs/reshape.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_reshape.pyx.c.o
      [90/151] Compiling C object pandas/_libs/writers.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_writers.pyx.c.o
      [91/151] Compiling C object pandas/_libs/tslibs/strptime.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_strptime.pyx.c.o
      [92/151] Compiling C object pandas/_libs/tslib.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslib.pyx.c.o
      [93/151] Compiling C object pandas/_libs/internals.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_internals.pyx.c.o
      [94/151] Compiling C object pandas/_libs/tslibs/parsing.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_parsing.pyx.c.o
      [95/151] Compiling C object pandas/_libs/tslibs/period.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_period.pyx.c.o
      [96/151] Compiling C object pandas/_libs/tslibs/timestamps.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_timestamps.pyx.c.o
      [97/151] Compiling C object pandas/_libs/parsers.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_parsers.pyx.c.o
      [98/151] Compiling C object pandas/_libs/tslibs/timedeltas.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_timedeltas.pyx.c.o
      [99/151] Compiling C object pandas/_libs/lib.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_lib.pyx.c.o
      [100/151] Compiling C object pandas/_libs/index.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_index.pyx.c.o
      [101/151] Compiling C object pandas/_libs/sparse.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_sparse.pyx.c.o
      [102/151] Compiling C object pandas/_libs/tslibs/offsets.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_offsets.pyx.c.o
      [103/151] Compiling C object pandas/_libs/interval.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_interval.pyx.c.o
      ninja: build stopped: subcommand failed.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
