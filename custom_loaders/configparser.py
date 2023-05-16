import configparser
from pathlib import Path
from typing import Iterator, List, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class ConfigLoader(BaseLoader):
    """
    A Config document loader that inherits from the BaseLoader class.

    This class can be initialized with either a single source file or a source
    directory containing .ini, .cfg, or .env files.
    """

    def __init__(self, source: Union[str, Path]):
        """Initialize the ConfigLoader with a source file or directory."""
        self.source = Path(source)

    def load(self) -> List[Document]:
        """Load and return all documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load the config documents from the source file or directory."""

        if self.source.is_file() and self.source.suffix in {".ini", ".cfg", ".env"}:
            files = [self.source]
        elif self.source.is_dir():
            files = list(self.source.glob("**/*.ini")) + list(self.source.glob("**/*.cfg")) + list(self.source.glob("**/*.env"))
        else:
            raise ValueError("Invalid source path or file type")

        for file_path in files:
            config = configparser.ConfigParser()
            config.read(file_path)

            doc = Document(
                page_content=str(dict(config._sections)),
                metadata={"source": str(file_path)},
            )
            yield doc
