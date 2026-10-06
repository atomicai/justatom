import simplejson as json


def delete_folder(pth):
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
        else:
            sub.unlink()
    pth.rmdir()


def io_snapshot(data, where=None, snapshot_number: str = "0", snapshot_prefix: str = None, snapshot_suffix: str = None):
    import os
    from pathlib import Path

    from loguru import logger

    where = Path(os.getcwd()) if not where else Path(where)
    snapshot_prefix = "" if snapshot_prefix is None else snapshot_prefix
    snapshot_suffix = "" if snapshot_suffix is None else snapshot_suffix
    filename = f"{snapshot_prefix}{str(snapshot_number)}{snapshot_suffix}.json"
    where_path = where / filename
    where.mkdir(parents=True, exist_ok=True)
    is_ok: bool = None
    try:
        with open(str(where_path), "w+") as fout:
            json.dump(data, fout, ensure_ascii=False)
    except:  # noqa
        logger.error(f"The data coming {data} is not JSON compliant")
        is_ok = False
    else:
        is_ok = True
    return is_ok


__all__ = ["delete_folder", "io_snapshot"]
