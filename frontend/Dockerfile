FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy frontend source code
COPY . .

EXPOSE 5173

# Launch Vite server bound to 0.0.0.0
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
