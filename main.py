import os
import shutil
import sys
import asyncio
import time

import uvicorn
from pathlib import Path
from src.adapters.http.router_api import router_api

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = router_api.app
app.include_router(router_api.router, tags=['restful'])


async def clean_tmp(dir_name):
    while True:
        path = os.fspath(Path(__file__).parent / dir_name)
        for dirpath, dirnames, filenames in os.walk(path):
            for dirname in dirnames:
                subdirectory_path = os.path.join(dirpath, dirname)
                subdirectory_mtime = os.stat(subdirectory_path).st_mtime
                if time.time() - subdirectory_mtime > 1800:
                    shutil.rmtree(subdirectory_path)
                    print(f'Deleted subdirectory {subdirectory_path}')
        await asyncio.sleep(1800)

@app.on_event('startup')
def startup_event():
    tmp_cleanup = asyncio.get_event_loop()
    tmp_cleanup.create_task(clean_tmp('tmp'))
    tmp_download_cleanup = asyncio.get_event_loop()
    tmp_download_cleanup.create_task(clean_tmp('tmp_download'))

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)
