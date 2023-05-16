import yaml
from pathlib import Path
from typing import Iterator, List, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class YAMLLoader(BaseLoader):
    """
    A YAML document loader that inherits from the BaseLoader class.

    This class can be initialized with either a single source file or a source
    directory containing .yaml files.
    """

    def __init__(self, source: Union[str, Path]):
        """Initialize the YAMLLoader with a source file or directory."""
        self.source = Path(source)

    def load(self) -> List[Document]:
        """Load and return all documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load the YAML documents from the source file or directory."""

        if self.source.is_file() and self.source.suffix == ".yaml":
            files = [self.source]
        elif self.source.is_dir():
            files = list(self.source.glob("**/*.yaml"))
        else:
            raise ValueError("Invalid source path or file type")

        for file_path in files:
            with open(file_path, 'r') as stream:
                try:
                    data = yaml.safe_load(stream)
                    doc = Document(
                        page_content=str(data),
                        metadata={"source": str(file_path)},
                    )
                    yield doc
                except yaml.YAMLError as exc:
                    print(exc)
