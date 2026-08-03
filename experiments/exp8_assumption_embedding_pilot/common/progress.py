from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        """Silent fallback; the project dependency installs the real tqdm."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "tqdm":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def update(self, amount: int = 1) -> None:
            pass


T = TypeVar("T")


def run_parallel(items: Iterable[T], function: Callable[[T], None], jobs: int, description: str) -> None:
    values = list(items)
    if not values:
        raise RuntimeError(f"{description} has no work units")
    with tqdm(total=len(values), desc=description, unit="task", dynamic_ncols=True) as progress:
        with ThreadPoolExecutor(max_workers=min(jobs, len(values))) as executor:
            futures = [executor.submit(function, value) for value in values]
            for future in as_completed(futures):
                future.result()
                progress.update(1)


def run_single(function: Callable[[], None], description: str) -> None:
    with tqdm(total=1, desc=description, unit="task", dynamic_ncols=True) as progress:
        function()
        progress.update(1)

