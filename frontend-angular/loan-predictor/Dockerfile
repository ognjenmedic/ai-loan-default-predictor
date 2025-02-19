# ----------------------------
# 1) BUILD STAGE
# ----------------------------
    FROM node:18-alpine AS build

    # Set non-interactive mode for npm/Angular
    ENV CI=true
    
    # Create app directory
    WORKDIR /app
    
    # Copy package files and install dependencies
    COPY package*.json ./
    RUN npm install
    
    # Copy the rest of your Angular code
    COPY . .
    
    # Build the Angular app (standard SPA build, not SSR)
    RUN npm run build -- --configuration production
    
    
    # ----------------------------
    # 2) SERVE STAGE
    # ----------------------------
    FROM node:18-alpine
    
    WORKDIR /app
    
    # Copy the built files from the build stage.
    COPY --from=build /app/dist/loan-predictor /app/dist
    
    # Install http-server globally
    RUN npm install -g http-server
    
    # Expose port 80
    EXPOSE 80
    
    # Run http-server at container start
    CMD ["http-server", "/app/dist", "-p", "80"]
    