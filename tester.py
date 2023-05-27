import aiohttp
import asyncio
import json


async def send_bytes_with_json():
    url = 'http://0.0.0.0:8585/predict'  # Replace with the desired URL

    headers = {
        'Content-Type': 'image/jpeg',

    }

    # Create a client session
    async with aiohttp.ClientSession() as session:
        a = [{'left': 232, 'top': 152, 'right': 342, 'bottom': 286},
             {'left': 413, 'top': 72, 'right': 530, 'bottom': 228},
             {'left': 660, 'top': 217, 'right': 718, 'bottom': 292},
             {'left': 405, 'top': 198, 'right': 438, 'bottom': 249}]

        # Send the request
        body={}
        async with session.post(url, headers=headers, json=body) as response:
            # Optionally, handle the response
            response_text = await response.text()
            print(response_text)


# Run the event loop
asyncio.run(send_bytes_with_json())
