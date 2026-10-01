# Frontend image — multi-stage build: Node for the Vite build, nginx to serve.
FROM node:24-alpine AS build

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit

COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine

# Replace the default nginx config: serve the SPA and proxy /api to the
# backend service (docker-compose service name).
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1
