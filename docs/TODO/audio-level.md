# One audio server for two machines: what is still open

> For a gig on 21 Nov 2026: one audio server (the Rust engine, on the Mac, started by WZRD)
> sends its OSC to WZRD on `127.0.0.1:9000` and to the LED controller `ledctl` on a Raspberry
> Pi. Sibling plans: `~/Documents/GitHub/animated_LED/docs/TODO/wzrd-input.md` (`ledctl`) and
> `~/Documents/GitHub/Gigabrain/VJ/WZRD/docs/TODO/led-sync.md` (WZRD).
>
> **Settled by Xander (2026-11-13), do not reopen:** the level's floor and time constant are
> fixed constants; the Pi's address is set by hand-editing the gitignored `configs/main.yaml`
> (no UI field for the gig); onsets stay single packets with no redundancy; the Python
> engine's resolver can follow the gig.

## Open: the gig runbook (Xander, with the Pi on)

1. Quit WZRD (it stops the server it started), or Ctrl-C a server started by hand. A running
   server rewrites `configs/main.yaml` from memory, so an edit made while it runs can be lost.
2. Add a second entry under `osc.destinations`: `- host: <pi-name>.local` / `port: 9000` (the
   port of `ledctl`'s audio listener). If the hotspot does not carry mDNS, use the Pi's IP.
3. Start WZRD. The server log (`audio-server.log` in WZRD's data folder) shows the resolve
   line; `ledctl`'s UI shows audio as connected. A config that fails to parse is moved aside
   and reseeded from the example, which drops the entry: check the log once after the edit.
4. Keep the FFT off (WZRD switches it off at start): `/audio/fft` goes to every destination,
   and at 1024 bins it is over 4 KB and fragments on WiFi.

## Open: the rehearsal check with the real Pi

Start WZRD with the Pi powered off, power the Pi on, and see `ledctl` report audio within
about 10 s of it joining the hotspot. Then play one loop at two volumes 20 dB apart:
`/audio/level` differs by 0.33 and stays apart, while `/audio/lmh` returns to its old range
within a minute. Not exercised on the Mac: a host whose address really changes, and the
resolver on Linux.

## Open, after the gig: the Python engine's resolver

The reference engine still needs literal IPs (`engine_contracts.md` says so): it resolves in
the `OscSender` constructor, so an unknown name stops the server from starting, and again on
every packet on the DSP worker. Same contract as the Rust engine, smallest change:
`OscSender` keeps `self._addrs` as a list of resolved `(ip, port)` tuples, replaced as a whole
by a daemon thread; the hot `send_*` methods read the attribute once per call. `send_meta`
stops using `SimpleUDPClient` (it resolves per send, on the asyncio loop): build the bytes
with pythonosc's `OscMessageBuilder` and `sendto` the same resolved list. *Done when* the
Python server starts with `no-such-host.local` in the list, and DSP load in `server_status`
is unchanged with `localhost` as a destination. No test for it.

## Open only if rehearsal shows missed LED flashes: onset redundancy

An onset is one packet, so a lost packet is a missed flash. If that shows: put a per-band
counter in the existing int argument, send each onset on its block and again on the next, and
have receivers ignore a counter they have seen. A copy sent in the same instant does not help
(WiFi loss comes in bursts of tens of milliseconds). It is a both-engines change plus the wire
goldens, and `ledctl` must de-duplicate. Not without evidence.

## Later

- An "OSC destinations" editor in the server's page, with per-destination resolve state in
  `server_status` (a WS message, a validator and UI in both engines).
- Per-destination filtering, or a lower rate for remote destinations.

## Shipped

`/audio/level` (a loudness on a fixed scale, never auto-gained) in both engines at parity,
with DSP and wire goldens; host-name destinations resolved on the `osc-resolve` thread in the
Rust engine; `configs/*.yaml` gitignored except the example. Nothing here depends on the
Rust-on-Pi phase of `rust_port.md`: the gig runs this server on the Mac only.

Done — see [reference/engine_contracts.md](../reference/engine_contracts.md) (the wire, the
level's definition, destinations) and [reference/rust_server.md](../reference/rust_server.md)
(the resolver thread).
