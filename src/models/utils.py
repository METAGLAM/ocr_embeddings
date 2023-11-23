from tensorboard.plugins import projector


def add_embedding_to_projector_config(
        config: projector.ProjectorConfig,
        tensor_name: str,
        metadata_path: str) -> None:
    """
    Given a Projector Config object add an embedding to the projection with
    the given name and metadata associated.

    Args:
        config: the Projector Config object
        tensor_name: the name of the tensor holding the embeddings
        metadata_path: the path to the metadata associated to the embeddings
    """
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = f'{tensor_name}/.ATTRIBUTES/VARIABLE_VALUE'
    embedding.metadata_path = metadata_path
