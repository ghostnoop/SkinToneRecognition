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

        body = {'link': 'https://358305.selcdn.ru/Asup_media/instagram/Cp-Fbe9opVf__1958692804.jpg',
                'coordinates': [{'left': 111, 'top': 70, 'right': 162, 'bottom': 139},
                                {'left': 65, 'top': 181, 'right': 116, 'bottom': 252}]}

        # Send the request
        async with session.post(url, headers=headers, json=body) as response:
            # Optionally, handle the response
            response_text = await response.text()
            print(response_text)


# Run the event loop
asyncio.run(send_bytes_with_json())
