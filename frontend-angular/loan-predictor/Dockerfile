# ----------------------------
# 1) BUILD STAGE
# ----------------------------
    FROM node:18-alpine AS build

    # Create app directory
    WORKDIR /app
    
    # Copy package files and install dependencies
    COPY package*.json ./
    RUN npm install
    
    # Copy the rest of your Angular code
    COPY . .
    
    # Build the Angular app
    RUN npm run build -- --configuration production
    
    
    # ----------------------------
    # 2) SERVE STAGE
    # ----------------------------
    FROM node:18-alpine
    
    # Create app directory
    WORKDIR /app
    
    # Copy the built files from the build stage.
    COPY --from=build /app/dist/loan-predictor /app/dist
    
    # Install http-server globally
    RUN npm install -g http-server
    
    # Expose port 80
    EXPOSE 80
    
    # Run http-server at container start
    CMD ["http-server", "/app/dist", "-p", "80"]
    