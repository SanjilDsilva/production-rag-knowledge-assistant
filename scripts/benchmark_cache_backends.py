from pathlib import Path
from statistics import quantiles
from tempfile import TemporaryDirectory
from time import perf_counter

from production_rag.answer_cache import create_cache_backend

ITERATIONS = 1000


def measure_backend(name, cache):
    cache.set("warm-key", "answer")
    get_times = []
    set_times = []

    for index in range(ITERATIONS):
        key = "warm-key" if index % 2 == 0 else f"missing-{index}"
        started = perf_counter()
        cache.get(key)
        get_times.append((perf_counter() - started) * 1000)

        started = perf_counter()
        cache.set("write-key", f"answer-{index}")
        set_times.append((perf_counter() - started) * 1000)

    stats = cache.stats()
    get_percentiles = quantiles(get_times, n=100)
    set_percentiles = quantiles(set_times, n=100)

    print(
        f"{name}: hits={stats['hits']}, misses={stats['misses']}, "
        f"hit_rate={stats['hit_rate']:.3f}, "
        f"get_p50_ms={get_percentiles[49]:.4f}, "
        f"get_p95_ms={get_percentiles[94]:.4f}, "
        f"set_p50_ms={set_percentiles[49]:.4f}, "
        f"set_p95_ms={set_percentiles[94]:.4f}"
    )


def main():
    with TemporaryDirectory() as temporary_directory:
        for backend_name in ("memory", "sqlite"):
            cache = create_cache_backend(
                backend_name,
                namespace="benchmark",
                max_entries=512,
                ttl_seconds=60,
                database_path=Path(temporary_directory) / f"{backend_name}.db",
            )
            measure_backend(backend_name, cache)


if __name__ == "__main__":
    main()
