SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"
DATA_DIR="$ROOT_DIR/data"
mkdir -p "$DATA_DIR"

# Download the wikidata data file if not exists
WIKIDATA_FILE="${DATA_DIR}/wikidata.vec"
WIKIDATA_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
if [ ! -f "$WIKIDATA_FILE" ]; then
    echo "$WIKIDATA_FILE does not exist. Downloading..."
    curl -o temporary.zip "$WIKIDATA_URL"
    unzip temporary.zip
    mv wiki-news-300d-1M.vec "$WIKIDATA_FILE"
    rm -rf temporary.zip
else
    echo "$WIKIDATA_FILE already exists."
fi

