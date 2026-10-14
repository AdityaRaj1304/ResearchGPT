# Multi-stage Dockerfile for ResearchGPT Frontend
# Stage 1: Build the production static bundle
FROM node:20-alpine AS build

WORKDIR /app

# Install npm dependencies using clean install
COPY package*.json ./
RUN npm ci

# Copy application source and build assets
COPY . .

# Build optimized production bundle
RUN npm run build

# Stage 2: Serve the compiled SPA through Nginx reverse proxy
FROM nginx:1.27-alpine

# Remove default nginx static html
RUN rm -rf /usr/share/nginx/html/*

# Copy built application assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration with reverse proxy definitions
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose standard HTTP port
EXPOSE 80

# Health check to ensure Nginx is serving the index page
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://localhost:80/ || exit 1

# Start Nginx in foreground mode
CMD ["nginx", "-g", "daemon off;"]
