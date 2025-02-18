# Use Node.js for building Angular app
FROM node:18-alpine AS build

WORKDIR /app

# Copy package.json & install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy source code
COPY . .

# Build Angular app
RUN npm run build --prod

# Use a lightweight web server to serve Angular
FROM node:18-alpine
WORKDIR /app
COPY --from=build /app/dist /app/dist
RUN npm install -g http-server

EXPOSE 80
CMD ["http-server", "/app/dist", "-p", "80"]
