import os
import sys

import uvicorn

from src.adapters.http.router_api import router_api

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = router_api.app
app.include_router(router_api.router, tags=['restful'])

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8080)
