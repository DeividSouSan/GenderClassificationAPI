import uvicorn

if __name__ == "__main__":
    config = uvicorn.Config("api.main:app", port=5000, reload=True)
    server = uvicorn.Server(config)
    server.run()