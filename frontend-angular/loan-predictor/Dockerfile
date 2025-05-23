# ---------------------------------------
# 1) BUILD STAGE - Build Angular App
# ---------------------------------------
    FROM node:18 AS build
    WORKDIR /usr/src/app
    
    # Copy package files and install dependencies
    COPY package.json package-lock.json ./
    RUN npm ci
    
    # Copy the entire Angular project
    COPY . .
    
    # Corrected build command
    RUN npm run build
    
    # ---------------------------------------
    # 2) SERVE STAGE - Serve with Nginx
    # ---------------------------------------
    FROM nginx:alpine
    WORKDIR /usr/share/nginx/html
    
    # Remove default Nginx static content
    RUN rm -rf ./*
    
    # Copy Angular build output to Nginx HTML directory
    COPY --from=build /usr/src/app/dist/loan-predictor /usr/share/nginx/html/ai-loan-default-predictor
    
    # Copy custom Nginx config
    COPY nginx.conf /etc/nginx/conf.d/default.conf
    
    # Expose port 80
    EXPOSE 80
    
    # Start Nginx
    CMD ["nginx", "-g", "daemon off;"]
    