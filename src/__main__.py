import uvicorn

import main

if __name__ == "__main__":
    app = main.create_app()
    uvicorn.run(app, host="0.0.0.0", port=9999)