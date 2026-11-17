.PHONY: install data experiment demo test lint clean

install:
	pip install -e ".[dev]"

data:
	timefence generate --customers 10000 --seed 7

experiment:
	timefence experiment

demo: data experiment
	timefence materialise --as-of "2025-12-31 23:00"
	timefence lookup C000042
	-timefence audit data/frames/event_time
	timefence audit data/frames/bitemporal

test:
	pytest

lint:
	ruff check src tests

clean:
	rm -rf data .pytest_cache
