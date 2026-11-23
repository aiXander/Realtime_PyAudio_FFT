.PHONY: start start-python build-rust check

# Two engines, one UI / config / OSC+WS interface. Both open the UI in the browser.
#   make start                  Rust engine (the default) on configs/main.yaml
#   make start-python           Python engine on configs/main.yaml
#   make start CONFIG=pi        either engine on configs/pi.yaml instead (works for start-python too)
#   make check                  prove the two engines agree (DSP goldens, Rust tests, live wire compare)
CONFIG ?= main

# main is the default config: pass no --config, so a missing main.yaml is
# seeded from main.example.yaml.
CONFIG_ARG := $(if $(filter main,$(CONFIG)),,--config configs/$(CONFIG).yaml)

build-rust:
	cd engine && cargo build --release -p server

start: build-rust
	engine/target/release/audio-server --open $(CONFIG_ARG)

start-python:
	uv run audio-server --open $(CONFIG_ARG)

# 1. regenerate the DSP goldens from the Python engine (the reference),
# 2. replay them in the Rust engine (cargo test includes the parity tests),
# 3. run both servers live on the BlackHole loopback and compare every WS/OSC
#    message shape (skipped with a message when BlackHole isn't installed).
check: build-rust
	uv run python tools/parity/make_goldens.py
	cd engine && cargo test --quiet
	uv run tools/parity/record_wire.py --engines
	@echo "make check: OK, the Python and Rust engines agree"
