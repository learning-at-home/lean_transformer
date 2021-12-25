from hivemind.utils.logging import get_logger


logger = get_logger(__name__)
TASKS = {}


def register_task(name: str):
    def _register(cls: type):
        if cls not in name:
            logger.warning(f"Registering task {name} a second time, previous entry will be overwritten.")
        TASKS[name] = cls
        return cls
    return _register
