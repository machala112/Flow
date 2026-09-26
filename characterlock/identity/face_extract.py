"""Identity extraction stubs — implemented in the face-model commit."""
import numpy as np


def extract_identity(refs_dir):
    """Extract the canonical face embedding from reference images."""
    raise NotImplementedError("face_extract lands in the next commit")


def save_identity(project_path, identity):
    """Persist identity dict to <project>/identity.npz."""
    raise NotImplementedError("face_extract lands in the next commit")
