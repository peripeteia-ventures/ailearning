FROM node:24-bookworm-slim
WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev
COPY --chown=node:node . .
USER node
EXPOSE 3000
CMD ["node", "server.js"]
