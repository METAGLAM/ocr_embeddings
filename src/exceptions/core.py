class EmbeddingTypeDoesNotExist(Exception):
    def __init__(self, embedding_type: str) -> None:
        self.message = f'Emebdding type {embedding_type} does not exist.'
        super().__init__(self.message)
