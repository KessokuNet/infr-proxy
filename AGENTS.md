# infr-proxy

- Commit each completed slice: once it builds and its tests pass, stage the files it touched and commit with a short message.
- When `~/.local/bin/infr-proxy` already exists, reinstall it from `target/release/infr-proxy` after building and restart `infr-proxy.service`.
