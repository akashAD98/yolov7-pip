import fire

from yolov7.detect import run as detect
from yolov7.models.common import export as export
from yolov7.train import run_cli as train
from yolov7.test import run as test


def app() -> None:
    """Cli app."""
    fire.Fire(
        {
            "train": train,
            "test": test,
            "detect": detect,
            "export": export,
        }
    )
