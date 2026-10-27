# syntax=docker/dockerfile:1.7
# One image: the Rust renderer and server with the built viewer embedded.
FROM node:24-slim AS viewer
RUN npm install -g pnpm@12.15.0 && npm cache clean --force
WORKDIR /src/viewer
COPY viewer/package.json viewer/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY viewer/ ./
RUN pnpm build

FROM rust:1.99.1-slim-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY src src
COPY --from=viewer /src/viewer/dist viewer/dist
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked && cp target/release/tealmap /tealmap

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data /config && chown 1000:1000 /data
COPY --from=build /tealmap /usr/local/bin/tealmap
USER 1000:1000
EXPOSE 8200
ENTRYPOINT ["tealmap", "--config", "/config/tealmap.toml"]
CMD ["serve"]
