"""Voice extraction stubs — implemented in the voice-model commit."""
import numpy as np


def extract_voice(voice_path):
    """Extract the speaker embedding from the reference audio."""
    raise NotImplementedError("voice_extract lands in the voice commit")


def save_voice_embedding(project_path, embedding):
    """Persist embedding to <project>/voice_embedding.npy."""
    raise NotImplementedError("voice_extract lands in the voice commit")
