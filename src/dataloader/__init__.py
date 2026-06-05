from .blender_reader_e2e import Blender6DDataset as Blender6DE2EDataset


def get_dataset(name: str, *args, **kwargs):
    if name == 'blender6d_e2e':
        return Blender6DE2EDataset(*args, **kwargs)
    raise ValueError(f"Dataset {name} not supported.")
