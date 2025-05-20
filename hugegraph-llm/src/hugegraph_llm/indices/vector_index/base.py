from abc import ABC, abstractmethod
from typing import List, Any, Union, Set


class VectorStoreBase(ABC):
    """
    Abstract base class defining the interface for a vector store.
    Implementations must support adding, removing, searching vectors,
    saving/loading from disk, and cleaning up resources.
    """

    @abstractmethod
    def add(self, vectors: List[List[float]], props: List[Any]):
        """
        Add a list of vectors and their corresponding properties to the store.

        Args:
            vectors (List[List[float]]): List of embedding vectors.
            props (List[Any]): List of associated metadata or properties for each vector.
        """

    @abstractmethod
    def remove(self, props: Union[Set[Any], List[Any]]) -> int:
        """
        Remove vectors based on their associated properties.

        Args:
            props (Union[Set[Any], List[Any]]): Properties of vectors to remove.

        Returns:
            int: Number of vectors removed.
        """

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int, dis_threshold: float = 0.9) -> List[Any]:
        """
        Search for the top_k most similar vectors to the query vector.

        Args:
            query_vector (List[float]): The vector to query against the index.
            top_k (int): Number of top results to return.
            dis_threshold (float): Distance threshold below which results are considered relevant.

        Returns:
            List[Any]: List of properties of the matched vectors.
        """

    @abstractmethod
    def to_index_file(self, dir_path: str):
        """
        Persist the vector store (index and metadata) to the specified directory.

        Args:
            dir_path (str): Path to the directory where the index and properties will be saved.
        """

    @staticmethod
    @abstractmethod
    def from_name(dir_path: str) -> "VectorStoreBase":
        """
        Load a vector store from the specified directory.

        Args:
            dir_path (str): Path to the directory containing the index and properties.

        Returns:
            VectorStore: An instance of the vector store.
        """

    @staticmethod
    @abstractmethod
    def clean(dir_path: str):
        """
        Delete the persisted index and properties from the specified directory.

        Args:
            dir_path (str): Path to the directory to clean.
        """
